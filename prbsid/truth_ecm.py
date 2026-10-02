from __future__ import annotations
import numpy as np
from scipy.signal import lfilter

class EcmCell:

    def __init__(self, r0=0.0008, r=(0.0002, 0.0003, 0.00025, 0.0004, 0.0006, 0.0009), tau=(0.001, 0.01, 0.1, 1.0, 10.0, 100.0), capacity_ah=60.0, ocv0=3.7, ocv_slope=0.7, r_scale=1.0, temp_c=25.0, soh=1.0):
        self.r0 = r0 * r_scale / soh
        self.r = np.asarray(r, float) * r_scale / soh
        self.tau = np.asarray(tau, float)
        self.capacity_ah = capacity_ah * soh
        self.ocv0, self.ocv_slope = (ocv0, ocv_slope)
        self.temp_c, self.soh = (temp_c, soh)
        self.kind = 'ecm'

    def ocv(self, soc):
        return self.ocv0 + self.ocv_slope * (np.asarray(soc, float) - 0.5)

    def eis_eta(self, freqs, soc=0.5):
        w = 2 * np.pi * np.asarray(freqs, float)
        return self.r0 + np.sum(self.r[None, :] / (1 + 1j * w[:, None] * self.tau[None, :]), axis=1)

    def steady_state(self, current):
        return self.r * current

    def run(self, current, dt, soc0, state0=None):
        i = np.asarray(current, float)
        n = i.size
        u0 = np.zeros(self.r.size) if state0 is None else np.asarray(state0, float)
        eta = self.r0 * i
        uend = np.empty(self.r.size)
        for m, (R, tau) in enumerate(zip(self.r, self.tau)):
            a = -np.expm1(-dt / tau)
            u, zf = lfilter([0.0, R * a], [1.0, -(1.0 - a)], i, zi=[(1.0 - a) * u0[m]])
            b = np.exp(-0.5 * dt / tau)
            eta = eta + b * u + (1.0 - b) * R * i
            uend[m] = (1.0 - a) * u[-1] + R * a * i[-1]
        q_c = np.cumsum(i) * dt - 0.5 * i * dt
        soc = soc0 - q_c / (self.capacity_ah * 3600.0)
        u_ocv = self.ocv(soc)
        return dict(v=u_ocv - eta, u_ocv=u_ocv, soc=soc, state=uend, soc_end=float(soc0 - i.sum() * dt / (self.capacity_ah * 3600.0)), complete=True)

class MismatchCell(EcmCell):

    def __init__(self, **kw):
        super().__init__(r0=0.00075, r=(0.00025, 0.00035, 0.0002, 0.0003, 0.00045, 0.0005, 0.0004, 0.0005), tau=(0.003, 0.02, 0.3, 2.5, 7.0, 35.0, 180.0, 700.0), **kw)
