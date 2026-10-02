from __future__ import annotations
import hashlib, json, os
from pathlib import Path
import numpy as np
from scipy.interpolate import PchipInterpolator
os.environ.setdefault('PYBAMM_DISABLE_TELEMETRY', 'true')
import pybamm
pybamm.set_logging_level('ERROR')
VARS = ('Voltage [V]', 'Bulk open-circuit voltage [V]', 'Discharge capacity [A.h]')

class PybammCell:
    kind = 'pybamm'

    def __init__(self, model: str='DFN', temp_c: float=25.0, soh: float=1.0, area_scale: float=1.0, capacity_ah: float=60.0, sei_fresh_m: float=5e-09, sei_growth_m_per_soh: float=3e-07, rtol: float=1e-08, atol: float=1e-10, cache_dir: str | os.PathLike | None=None, parameter_set: str='Chen2020'):
        self.model_name, self.temp_c, self.soh, self.area_scale = (model, float(temp_c), float(soh), float(area_scale))
        self.rtol, self.atol = (rtol, atol)
        self.options = {'surface form': 'differential', 'SEI': 'constant', 'thermal': 'isothermal'}
        p = pybamm.ParameterValues(parameter_set)
        base = float(p['Nominal cell capacity [A.h]'])
        p['Number of electrodes connected in parallel to make a cell'] = capacity_ah / base * area_scale
        p['Nominal cell capacity [A.h]'] = capacity_ah * area_scale * soh
        p['Negative electrode active material volume fraction'] = float(p['Negative electrode active material volume fraction']) * soh
        p['Positive electrode active material volume fraction'] = float(p['Positive electrode active material volume fraction']) * soh
        self.sei_m = sei_fresh_m + sei_growth_m_per_soh * (1.0 - soh)
        p['Initial SEI thickness [m]'] = self.sei_m
        p['Ambient temperature [K]'] = 273.15 + temp_c
        p['Initial temperature [K]'] = 273.15 + temp_c
        p['Lower voltage cut-off [V]'] = 2.5
        p['Upper voltage cut-off [V]'] = 4.3
        self._p = p
        self._sim = None
        self._cache_dir = Path(cache_dir) if cache_dir else None
        self._ocv = None
        self.meta = dict(model=model, parameter_set=parameter_set, temp_c=temp_c, soh=soh, area_scale=area_scale, capacity_ah=capacity_ah, sei_m=self.sei_m, options=self.options, rtol=rtol, atol=atol, pybamm=pybamm.__version__)

    def _model(self, name=None):
        return getattr(pybamm.lithium_ion, name or self.model_name)(options=dict(self.options))

    def _solver(self):
        return pybamm.IDAKLUSolver(rtol=self.rtol, atol=self.atol)

    def _ocv_table(self):
        if self._ocv is not None:
            return self._ocv
        key = hashlib.sha256(json.dumps(dict(s=self.soh, a=self.area_scale, sei=self.sei_m, c=float(self._p['Nominal cell capacity [A.h]']), v=pybamm.__version__), sort_keys=True).encode()).hexdigest()[:16]
        f = self._cache_dir / f'ocv_{key}.npz' if self._cache_dir else None
        if f is not None and f.exists():
            try:
                z = np.load(f)
                self._ocv = (PchipInterpolator(z['soc'], z['u']), float(z['q_th']))
                return self._ocv
            except Exception:
                pass
        p = self._p.copy()
        p['Ambient temperature [K]'] = 298.15
        p['Initial temperature [K]'] = 298.15
        p['Lower voltage cut-off [V]'] = 2.0
        cap = float(p['Nominal cell capacity [A.h]'])
        p['Current function [A]'] = cap / 20.0
        sim = pybamm.Simulation(self._model('SPM'), parameter_values=p, solver=pybamm.IDAKLUSolver(rtol=1e-08, atol=1e-10))
        sol = sim.solve([0, 22 * 3600.0], initial_soc=1.0, t_interp=np.linspace(0, 22 * 3600.0, 8801)[1:-1])
        q = np.asarray(sol['Discharge capacity [A.h]'].data, float)
        u = np.asarray(sol['Bulk open-circuit voltage [V]'].data, float)
        keep = np.concatenate([[True], np.diff(q) > 1e-09])
        q, u = (q[keep], u[keep])

        def rest_u(soc):
            pr = self._p.copy()
            pr['Current function [A]'] = 0.0
            s = pybamm.Simulation(self._model('SPM'), parameter_values=pr, solver=pybamm.IDAKLUSolver())
            return float(np.asarray(s.solve([0, 1.0], initial_soc=soc)['Bulk open-circuit voltage [V]'].data)[0])
        inv = PchipInterpolator(u[::-1], q[::-1])
        qa, qb = (float(inv(rest_u(0.9))), float(inv(rest_u(0.1))))
        q_th = (qb - qa) / 0.8
        soc = 0.9 - (q - qa) / q_th
        order = np.argsort(soc)
        self._ocv = (PchipInterpolator(soc[order], u[order]), q_th)
        if f is not None:
            try:
                self._cache_dir.mkdir(parents=True, exist_ok=True)
                tmp = f.with_name(f.stem + f'.{os.getpid()}.tmp.npz')
                np.savez(tmp, soc=soc[order], u=u[order], q_th=q_th)
                os.replace(tmp, f)
            except Exception:
                pass
        return self._ocv

    def ocv(self, soc):
        return self._ocv_table()[0](np.asarray(soc, float))

    @property
    def capacity_ah(self) -> float:
        return self._ocv_table()[1]

    def eis_total(self, freqs, soc: float) -> np.ndarray:
        eis = pybamm.EISSimulation(self._model(), parameter_values=self._p.copy())
        sol = eis.solve(np.asarray(freqs, float), initial_soc=float(soc))
        return np.asarray(sol['Impedance [Ohm]'].data).astype(complex).reshape(-1)

    def eis_eta(self, freqs, soc: float) -> np.ndarray:
        f = np.asarray(freqs, float)
        interp, q_th = self._ocv_table()
        du_dq = -float(interp.derivative()(soc)) / (q_th * 3600.0)
        return self.eis_total(f, soc) + du_dq / (1j * 2 * np.pi * f)

    def run(self, current, dt: float, soc0: float, state0=None) -> dict:
        i = np.asarray(current, float)
        n = i.size
        edges = np.flatnonzero(np.diff(i) != 0) + 1
        starts = np.concatenate([[0], edges])
        stops = np.concatenate([edges, [n]])
        if state0 is None:
            p = self._p.copy()
            p['Current function [A]'] = '[input]'
            self._sim = pybamm.Simulation(self._model(), parameter_values=p, solver=self._solver())
            self._sim.build(initial_soc=float(soc0))
            prev = None
            q0 = 0.0
        else:
            prev, q0 = state0
        sim = self._sim
        v = np.full(n, np.nan)
        u = np.full(n, np.nan)
        complete = True
        q_end = q0
        for a, b in zip(starts, stops):
            m = b - a
            mids = (np.arange(m) + 0.5) * dt
            kw = dict(dt=m * dt, t_eval=np.array([0.0, m * dt]), t_interp=mids, inputs={'Current function [A]': float(i[a])}, save=False)
            if prev is not None:
                kw['starting_solution'] = prev
            try:
                sol = sim.step(**kw)
            except Exception as exc:
                complete = False
                self.last_error = repr(exc)[:300]
                break
            if sol.termination != 'final time' or len(sol.t) != m + 2:
                complete = False
                self.last_error = f'terminated: {sol.termination}'
                break
            v[a:b] = np.asarray(sol[VARS[0]].data, float)[1:-1]
            u[a:b] = np.asarray(sol[VARS[1]].data, float)[1:-1]
            q_end = float(np.asarray(sol[VARS[2]].data, float)[-1])
            prev = sol
        interp, q_th = self._ocv_table()
        return dict(v=v, u_ocv=u, complete=complete, state=(prev, q_end), q_dis_ah=q_end, soc_end=float(soc0 - (q_end - q0) / q_th) if state0 is None else float('nan'))
