# scripts/render_overlays.py

"""
Render simple transparent PNG overlays (one per variable/step) from GRIB files.

Keeps dependencies light (no GDAL/cartopy): we draw on a regular lon/lat grid over the Fennoscandia bbox.
For real production, you may want XYZ tiles and nicer palettes/contours.
"""

import json, math, pathlib, glob
import numpy as np
import xarray as xr
from PIL import Image

DATA = pathlib.Path("data_raw")
OUT  = pathlib.Path("out"); OUT.mkdir(parents=True, exist_ok=True)

# Fennoscandia bbox (W,S,E,N)
W, S, E, N = 5.0, 55.0, 35.5, 72.5

# Lon/lat grid for rendering (Plate Carrée)
NX, NY = 900, 600  # ~0.034° res; tune later

# Steps we will export (strings with +hh formatting)
STEPS = ["+000","+006","+012","+024"]

# Available variables for the front-end
VARS = ["msl","t2m","tp_rate","tcwv","ssr_flux","str_flux","gh500","gh850","wspd850"]

def linspace2d():
    lons = np.linspace(W, E, NX, dtype=np.float32)
    lats = np.linspace(N, S, NY, dtype=np.float32)
    return np.meshgrid(lons, lats)  # lon2, lat2

LON2, LAT2 = linspace2d()

def save_png(name, rgba):
    img = Image.fromarray(rgba, mode="RGBA")
    img.save(OUT / f"{name}.png", optimize=True)

def colorize(data, vmin, vmax, base=(0,0,0), top=(0,180,255), alpha=180):
    arr = data.copy()
    mask = ~np.isfinite(arr)
    t = (arr - vmin) / (vmax - vmin + 1e-9)
    t = np.clip(t, 0, 1)
    r = (base[0]*(1-t) + top[0]*t).astype(np.uint8)
    g = (base[1]*(1-t) + top[1]*t).astype(np.uint8)
    b = (base[2]*(1-t) + top[2]*t).astype(np.uint8)
    a = np.full_like(r, alpha, dtype=np.uint8)
    a[mask] = 0
    rgba = np.dstack([r,g,b,a])
    return rgba

def to_grid(da):
    # Nearest neighbor resample da(lat,lon) to our grid
    latn = [n for n in da.dims if "lat" in n][0]
    lonn = [n for n in da.dims if "lon" in n][0]
    da = da.rename({latn:"lat", lonn:"lon"})
    # indexers must be 1D; we take first column for lat and first row for lon
    targ_lat = xr.DataArray(LAT2[:,0], dims=("y",))
    targ_lon = xr.DataArray(LON2[0,:], dims=("x",))
    res = da.sel(lat=targ_lat, lon=targ_lon, method="nearest").transpose("y","x").values
    return res

def open_cf(path):
    try:
        return xr.open_dataset(path, engine="cfgrib")
    except Exception as e:
        print(f"[WARN] open fail {path}: {e}")
        return None

def find_one(pattern):
    files = sorted(glob.glob(str(DATA / pattern)))
    return files[0] if files else None

def render_placeholders():
    print("[INFO] Rendering placeholder overlays...")
    base = np.full((NY, NX), np.nan, dtype=np.float32)
    for name, (vmin, vmax) in {
        "msl_+000": (960, 1040), "msl_+006": (960, 1040),
        "t2m_+000": (-25, 25), "t2m_+006": (-25, 25),
        "tp_rate_+006": (0, 6),
        "gh500_+000": (4800, 5800), "gh850_+000": (1200, 1700),
        "wspd850_+000": (0, 40),
        "ssr_flux_+006": (0, 500), "str_flux_+006": (-150, 50),
    }.items():
        rgba = colorize(base, * (vmin, vmax))
        save_png(name, rgba)

def main():
    # Discover raw files (just by simple glob patterns)
    msl = find_one("msl_sfc_*.grib2")
    t2  = find_one("2t_sfc_*.grib2")
    tp  = find_one("tp_sfc_*.grib2")
    ssr = find_one("ssr_sfc_*.grib2")
    strn= find_one("str_sfc_*.grib2")
    tcwv= find_one("tcwv_sfc_*.grib2")

    gh500= find_one("gh_pl_500_*.grib2")
    gh850= find_one("gh_pl_850_*.grib2")
    u850 = find_one("u_pl_850_*.grib2")
    v850 = find_one("v_pl_850_*.grib2")

    have_any = any([msl,t2,tp,ssr,strn,tcwv,gh500,gh850,u850,v850])
    if not have_any:
        render_placeholders()
        run_id = "latest"
    else:
        # Open what we have and render a minimal set safely
        run_id = "latest"
        # MSLP (assume variable name is 'msl' in Pa with time dimension 'time' or 'valid_time')
        if msl:
            ds = open_cf(msl)
            if ds is not None:
                vname = [n for n in ds.data_vars if n.startswith("msl")][0]
                time_dim = list(ds[vname].dims)[0]
                times = ds[time_dim].values
                # Render +000 and +006 if available
                for step_idx, step_tag in [(0,"+000"), (1,"+006")]:
                    if step_idx < ds[vname].shape[0]:
                        arr = ds[vname].isel({time_dim: step_idx}).sel(longitude=slice(W,E), latitude=slice(N,S))
                        # convert Pa to hPa
                        grid = to_grid(arr) / 100.0
                        rgba = colorize(grid, 960, 1040)
                        save_png(f"msl_{step_tag}", rgba)

        # 2m T (K) -> °C
        if t2:
            ds = open_cf(t2)
            if ds is not None:
                vname = [n for n in ds.data_vars if n.startswith("t2") or n=="2t"][0]
                time_dim = list(ds[vname].dims)[0]
                for step_idx, step_tag in [(0,"+000"), (1,"+006")]:
                    if step_idx < ds[vname].shape[0]:
                        arr = ds[vname].isel({time_dim: step_idx}).sel(longitude=slice(W,E), latitude=slice(N,S))
                        grid = to_grid(arr) - 273.15
                        rgba = colorize(grid, -25, 25)
                        save_png(f"t2m_{step_tag}", rgba)

        # tp (m acc) -> mm/h rate between 0 and +6
        if tp:
            ds = open_cf(tp)
            if ds is not None and ds.indexes:
                vname = [n for n in ds.data_vars if n.startswith("tp")][0]
                time_dim = list(ds[vname].dims)[0]
                if ds[vname].shape[0] >= 2:
                    a0 = ds[vname].isel({time_dim:0}).sel(longitude=slice(W,E), latitude=slice(N,S))
                    a1 = ds[vname].isel({time_dim:1}).sel(longitude=slice(W,E), latitude=slice(N,S))
                    # assume 6-hour window between steps 0->1 (safe for MVP)
                    grid = (to_grid(a1) - to_grid(a0)) * 1000.0 * (3600.0 / (6*3600.0))
                    rgba = colorize(grid, 0, 6, base=(0,0,0), top=(0,120,255))
                    save_png("tp_rate_+006", rgba)

        # gh 500/850 (m^2/s^2) -> m
        if gh500:
            ds = open_cf(gh500)
            if ds is not None:
                vname = [n for n in ds.data_vars if n.startswith("gh")][0]
                time_dim = list(ds[vname].dims)[0]
                a = ds[vname].isel({time_dim:0}).sel(longitude=slice(W,E), latitude=slice(N,S))
                grid = to_grid(a) / 9.80665
                rgba = colorize(grid, 4800, 5800)
                save_png("gh500_+000", rgba)

        if gh850:
            ds = open_cf(gh850)
            if ds is not None:
                vname = [n for n in ds.data_vars if n.startswith("gh")][0]
                time_dim = list(ds[vname].dims)[0]
                a = ds[vname].isel({time_dim:0}).sel(longitude=slice(W,E), latitude=slice(N,S))
                grid = to_grid(a) / 9.80665
                rgba = colorize(grid, 1200, 1700)
                save_png("gh850_+000", rgba)

        # 850-hPa wind speed from u,v
        if u850 and v850:
            dsu = open_cf(u850); dsv = open_cf(v850)
            if dsu is not None and dsv is not None:
                vn_u = [n for n in dsu.data_vars if n.startswith("u")][0]
                vn_v = [n for n in dsv.data_vars if n.startswith("v")][0]
                tu = list(dsu[vn_u].dims)[0]
                tv = list(dsv[vn_v].dims)[0]
                a = dsu[vn_u].isel({tu:0}).sel(longitude=slice(W,E), latitude=slice(N,S))
                b = dsv[vn_v].isel({tv:0}).sel(longitude=slice(W,E), latitude=slice(N,S))
                uu = to_grid(a); vv = to_grid(b)
                wspd = np.sqrt(uu*uu + vv*vv)
                rgba = colorize(wspd, 0, 40, base=(0,0,0), top=(255,255,255))
                save_png("wspd850_+000", rgba)

        # tcwv (kg/m^2)
        if tcwv:
            ds = open_cf(tcwv)
            if ds is not None:
                vname = [n for n in ds.data_vars if n.startswith("tcwv")][0]
                time_dim = list(ds[vname].dims)[0]
                a = ds[vname].isel({time_dim:0}).sel(longitude=slice(W,E), latitude=slice(N,S))
                grid = to_grid(a)
                rgba = colorize(grid, 0, 60, base=(0,0,0), top=(0,255,180))
                save_png("tcwv_+000", rgba)

        # ssr/str flux from accum (J/m^2) -> W/m^2 over first 6h
        if ssr:
            ds = open_cf(ssr)
            if ds is not None and ds[list(ds.data_vars)[0]].shape[0] >= 2:
                vname = [n for n in ds.data_vars if n.startswith("ssr")][0]
                time_dim = list(ds[vname].dims)[0]
                a0 = ds[vname].isel({time_dim:0}).sel(longitude=slice(W,E), latitude=slice(N,S))
                a1 = ds[vname].isel({time_dim:1}).sel(longitude=slice(W,E), latitude=slice(N,S))
                flux = (to_grid(a1) - to_grid(a0)) / (6*3600.0)
                rgba = colorize(flux, 0, 500, base=(0,0,0), top=(255,220,120))
                save_png("ssr_flux_+006", rgba)

        if strn:
            ds = open_cf(strn)
            if ds is not None and ds[list(ds.data_vars)[0]].shape[0] >= 2:
                vname = [n for n in ds.data_vars if n.startswith("str")][0]
                time_dim = list(ds[vname].dims)[0]
                a0 = ds[vname].isel({time_dim:0}).sel(longitude=slice(W,E), latitude=slice(N,S))
                a1 = ds[vname].isel({time_dim:1}).sel(longitude=slice(W,E), latitude=slice(N,S))
                flux = (to_grid(a1) - to_grid(a0)) / (6*3600.0)
                rgba = colorize(flux, -150, 50, base=(0,0,0), top=(255,120,120))
                save_png("str_flux_+006", rgba)

    # Write manifest (even for placeholders)
    manifest = {
        "run": run_id,
        "bbox": [W, S, E, N],
        "steps": STEPS,
        "vars": VARS
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("[OK] Overlays written to", OUT)

if __name__ == "__main__":
    main()
