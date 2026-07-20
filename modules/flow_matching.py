from abc import ABC

import torch
import torch.nn.functional as F

from modules.diffusion_transformer import DiT
from modules.commons import sequence_mask

from tqdm import tqdm

class BASECFM(torch.nn.Module, ABC):
    def __init__(
        self,
        args,
    ):
        super().__init__()
        self.sigma_min = 1e-6

        self.estimator = None

        self.in_channels = args.DiT.in_channels

        self.criterion = torch.nn.MSELoss() if args.reg_loss_type == "l2" else torch.nn.L1Loss()

        if hasattr(args.DiT, 'zero_prompt_speech_token'):
            self.zero_prompt_speech_token = args.DiT.zero_prompt_speech_token
        else:
            self.zero_prompt_speech_token = False

    @torch.inference_mode()
    def inference(self, mu, x_lens, prompt, style, f0, n_timesteps, temperature=1.0, inference_cfg_rate=0.5, sway_sampling_coef=0.0, ode_method="euler", cfg_rescale=0.0):
        """Forward diffusion

        Args:
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            n_timesteps (int): number of diffusion steps
            temperature (float, optional): temperature for scaling noise. Defaults to 1.0.
            spks (torch.Tensor, optional): speaker ids. Defaults to None.
                shape: (batch_size, spk_emb_dim)
            cond: Not used but kept for future purposes

        Returns:
            sample: generated mel-spectrogram
                shape: (batch_size, n_feats, mel_timesteps)
        """
        B, T = mu.size(0), mu.size(1)
        z = torch.randn([B, self.in_channels, T], device=mu.device) * temperature
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device)
        if sway_sampling_coef != 0.0:
            t_span = t_span + sway_sampling_coef * (torch.cos(torch.pi / 2 * t_span) - 1 + t_span)
        return self.solve_euler(z, x_lens, prompt, mu, style, f0, t_span, inference_cfg_rate, ode_method, cfg_rescale)

    def solve_euler(self, x, x_lens, prompt, mu, style, f0, t_span, inference_cfg_rate=0.5, ode_method="euler", cfg_rescale=0.0):
        """
        Fixed-grid (euler / midpoint / rk4) or adaptive (rk45) ODE solver for
        the conditional flow-matching trajectory.

        ode_method:
            "euler"   - 1st order, original behavior (default)
            "midpoint"- 2nd order RK2
            "rk4"     - 4th order Runge-Kutta, cleaner mel at equal step count
            "rk45"    - adaptive embedded Runge-Kutta-Fehlberg 4(5); automatically
                        allocates steps where the trajectory is curved. Best
                        quality for a given step budget. Ignores sway resampling.
        cfg_rescale: SDXL-style guidance rescale (0.0 = disabled / prior behavior).
            When > 0 it pulls the CFG-combined velocity back toward the
            conditional velocity distribution, reducing over-saturation /
            metallic coloration at high inference_cfg_rate.
        """
        t_start, _, _ = t_span[0], t_span[-1], t_span[1] - t_span[0]

        # I am storing this because I can later plot it by putting a debugger here and saving it to a file
        # Or in future might add like a return_all_steps flag
        sol = []
        # apply prompt
        prompt_len = prompt.size(-1)
        prompt_x = torch.zeros_like(x)
        prompt_x[..., :prompt_len] = prompt[..., :prompt_len]
        x[..., :prompt_len] = 0
        if self.zero_prompt_speech_token:
            mu[..., :prompt_len] = 0

        def eval_dphi_dt(x_in, t_in):
            # Zero the prompt region of the ODE state for every evaluation so
            # that RK2/RK4/RK45 intermediate states stay consistent with the
            # original euler step (which re-zeroed x after every update).
            x_in = x_in.clone()
            x_in[..., :prompt_len] = 0.0
            if inference_cfg_rate > 0:
                stacked_prompt_x = torch.cat([prompt_x, torch.zeros_like(prompt_x)], dim=0)
                stacked_style = torch.cat([style, torch.zeros_like(style)], dim=0)
                stacked_mu = torch.cat([mu, torch.zeros_like(mu)], dim=0)
                stacked_x = torch.cat([x_in, x_in], dim=0)
                stacked_t = torch.cat([t_in.unsqueeze(0), t_in.unsqueeze(0)], dim=0)

                stacked_dphi_dt = self.estimator(
                    stacked_x, stacked_prompt_x, x_lens, stacked_t, stacked_style, stacked_mu,
                )

                d_cond, d_uncond = stacked_dphi_dt.chunk(2, dim=0)
                dphi_dt = (1.0 + inference_cfg_rate) * d_cond - inference_cfg_rate * d_uncond
            else:
                d_cond = self.estimator(x_in, prompt_x, x_lens, t_in.unsqueeze(0), style, mu)
                dphi_dt = d_cond
            if cfg_rescale > 0.0:
                std_cond = d_cond.std()
                std_phi = dphi_dt.std()
                if std_phi > 1e-8 and std_cond > 1e-8:
                    x_reshaped = dphi_dt * (std_cond / std_phi)
                    dphi_dt = cfg_rescale * x_reshaped + (1.0 - cfg_rescale) * dphi_dt
            return dphi_dt

        if ode_method == "rk45":
            # ---- Adaptive embedded Runge-Kutta-Fehlberg 4(5) ----
            tol = 1e-3
            max_steps = max(int(t_span.numel()) - 1, 1)
            t = float(t_start)
            dt = 1.0 / max(max_steps, 1)
            steps = 0
            while t < 1.0 - 1e-12 and steps < max_steps:
                dt = min(dt, 1.0 - t)
                t_t = torch.tensor(t, device=mu.device)
                dt_t = torch.tensor(dt, device=mu.device)
                k1 = eval_dphi_dt(x, t_t)
                k2 = eval_dphi_dt(x + dt_t / 4 * k1, t_t + dt_t / 4)
                k3 = eval_dphi_dt(x + dt_t * (3.0 / 32 * k1 + 9.0 / 32 * k2), t_t + 3 * dt_t / 8)
                k4 = eval_dphi_dt(x + dt_t * (1932.0 / 2197 * k1 - 7200.0 / 2197 * k2 + 7296.0 / 2197 * k3), t_t + 12 * dt_t / 13)
                k5 = eval_dphi_dt(x + dt_t * (439.0 / 216 * k1 - 8 * k2 + 3680.0 / 513 * k3 - 845.0 / 4104 * k4), t_t + dt_t)
                k6 = eval_dphi_dt(x + dt_t * (-8.0 / 27 * k1 + 2 * k2 - 3544.0 / 2565 * k3 + 1859.0 / 4104 * k4 - 11.0 / 40 * k5), t_t + dt_t / 2)
                y4 = x + dt_t * (25.0 / 216 * k1 + 1408.0 / 2565 * k3 + 2197.0 / 4104 * k4 - 1.0 / 5 * k5)
                y5 = x + dt_t * (16.0 / 135 * k1 + 6656.0 / 12825 * k3 + 28561.0 / 56430 * k4 - 9.0 / 50 * k5 + 2.0 / 55 * k6)
                diff = torch.abs(y5 - y4)
                scale = torch.abs(y5) + torch.abs(x) + 1e-3
                err = float(torch.max(diff / scale))
                if err < 1e-12:
                    factor = 4.0
                    accept = True
                else:
                    factor = 0.9 * (tol / err) ** 0.2
                    factor = min(max(factor, 0.2), 4.0)
                    accept = err <= tol
                if accept:
                    x = y5
                    x[..., :prompt_len] = 0
                    t = t + dt
                    steps += 1
                    sol.append(x)
                    dt = dt * factor
                else:
                    dt = dt * factor
                if dt < 1e-4:
                    dt = 1e-4
            if t < 1.0 - 1e-9:
                # Force the remaining trajectory to t=1 with small fixed steps.
                remain = 1.0 - t
                n = max(1, int(round(remain / 0.02)))
                for _ in range(n):
                    dtf = (1.0 - t) / n
                    d = eval_dphi_dt(x, torch.tensor(t, device=mu.device))
                    x = x + torch.tensor(dtf, device=mu.device) * d
                    x[..., :prompt_len] = 0
                    t = t + dtf
                sol.append(x)
            return sol[-1]

        # ---- Fixed-grid solvers: euler / midpoint / rk4 ----
        for step in tqdm(range(1, len(t_span))):
            dt = t_span[step] - t_span[step - 1]
            t = t_span[step - 1]
            if ode_method == "rk4":
                k1 = eval_dphi_dt(x, t)
                k2 = eval_dphi_dt(x + 0.5 * dt * k1, t + 0.5 * dt)
                k3 = eval_dphi_dt(x + 0.5 * dt * k2, t + 0.5 * dt)
                k4 = eval_dphi_dt(x + dt * k3, t + dt)
                dphi_dt = (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
            elif ode_method == "midpoint":
                k1 = eval_dphi_dt(x, t)
                x_mid = x + 0.5 * dt * k1
                x_mid[:, :, :prompt_len] = 0
                k2 = eval_dphi_dt(x_mid, t + 0.5 * dt)
                dphi_dt = k2
            else:
                dphi_dt = eval_dphi_dt(x, t)

            x = x + dt * dphi_dt
            t = t + dt
            sol.append(x)
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t
            x[:, :, :prompt_len] = 0

        return sol[-1]
    def forward(self, x1, x_lens, prompt_lens, mu, style):
        """Computes diffusion loss

        Args:
            x1 (torch.Tensor): Target
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): target mask
                shape: (batch_size, 1, mel_timesteps)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            spks (torch.Tensor, optional): speaker embedding. Defaults to None.
                shape: (batch_size, spk_emb_dim)

        Returns:
            loss: conditional flow matching loss
            y: conditional flow
                shape: (batch_size, n_feats, mel_timesteps)
        """
        b, _, t = x1.shape

        # random timestep
        t = torch.rand([b, 1, 1], device=mu.device, dtype=x1.dtype)
        # sample noise p(x_0)
        z = torch.randn_like(x1)

        y = (1 - (1 - self.sigma_min) * t) * z + t * x1
        u = x1 - (1 - self.sigma_min) * z

        prompt = torch.zeros_like(x1)
        for bib in range(b):
            prompt[bib, :, :prompt_lens[bib]] = x1[bib, :, :prompt_lens[bib]]
            # range covered by prompt are set to 0
            y[bib, :, :prompt_lens[bib]] = 0
            if self.zero_prompt_speech_token:
                mu[bib, :, :prompt_lens[bib]] = 0

        estimator_out = self.estimator(y, prompt, x_lens, t.squeeze(1).squeeze(1), style, mu, prompt_lens)
        loss = 0
        for bib in range(b):
            loss += self.criterion(estimator_out[bib, :, prompt_lens[bib]:x_lens[bib]], u[bib, :, prompt_lens[bib]:x_lens[bib]])
        loss /= b

        return loss, estimator_out + (1 - self.sigma_min) * z



class CFM(BASECFM):
    def __init__(self, args):
        super().__init__(
            args
        )
        if args.dit_type == "DiT":
            self.estimator = DiT(args)
        else:
            raise NotImplementedError(f"Unknown diffusion type {args.dit_type}")
