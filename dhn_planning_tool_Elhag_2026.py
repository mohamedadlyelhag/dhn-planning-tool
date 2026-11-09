"""
District Heating Network Planner - Streamlit Application

This application allows users to:
1. Select a geographic area and extract building footprints from OpenStreetMap
2. Calculate heat demand for buildings based on type and area
3. Design a district heating network by placing pipes and connecting buildings
4. Analyze network performance including flow rates, pressure losses, and heat losses
5. Generate reports with network data

The application uses:
- OpenStreetMap for geographic data
- Streamlit for the web interface
- Folium for interactive maps
- GeoPandas for spatial operations
- NetworkX for graph-based network modeling
"""
# ============================================================
# SECTION: Imports
# ============================================================

import osmnx as ox
import geopandas as gpd
import numpy as np
import folium
from folium.plugins import Draw, MeasureControl, PolyLineTextPath, Fullscreen, MiniMap, MousePosition
import streamlit as st
import pandas as pd
from shapely.geometry import shape, Point, LineString, Polygon, mapping
from shapely.ops import split, snap, nearest_points
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim
from sklearn.cluster import DBSCAN
from io import BytesIO
from collections import defaultdict
from networkx import DiGraph, dfs_preorder_nodes, shortest_path
import networkx as nx
import math
import re
import json
import threading
from branca.colormap import LinearColormap
from branca.element import MacroElement, Template


# ============================================================
# SECTION: Config / Constants
# ============================================================

# Standard pipe diameters in mm
DN_SIZES = [10, 15, 20, 25, 32, 40, 50, 65, 80, 100, 125, 150, 200, 250, 300,
            350, 400, 450, 500, 600, 700, 800, 900, 1000]

geolocator = Nominatim(user_agent="dhn-planner")

# Demand profiles for different building types.
# Each profile contains:
#   space_heating: annual heating demand per m2 (kWh/m2/year)
#   dhw:           domestic hot water demand per m2 (kWh/m2/year)
#   color:         display color for this building type
#   simultaneity_factor: peak demand factor (0-1)
#   daily_profile: 24-hour demand profile (relative values)
DEMAND_PROFILES = {
    'Residential': {
        'space_heating': 100,   # kWh/m2/year
        'dhw': 20,              # kWh/m2/year
        'color': 'blue',
        'simultaneity_factor': 0.7,
        'peak_to_average': 3.7,      # design peak ≈ 3.7 × annual average
        'peak_safety_factor': 1.2,   # load uncertainty / future growth margin (VDI 4713-1)
        'daily_profile': [0.3, 0.2, 0.1, 0.1, 0.2, 0.5, 0.8, 0.9, 0.7, 0.6, 0.5, 0.5,
                          0.6, 0.7, 0.8, 0.9, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
    },
    'Office': {
        'space_heating': 90, 'dhw': 15, 'color': 'green', 'simultaneity_factor': 0.8,
        'daily_profile': [0.1, 0.1, 0.1, 0.1, 0.2, 0.4, 0.8, 1.0, 0.9, 0.8, 0.7, 0.7,
                          0.7, 0.7, 0.8, 0.9, 0.9, 0.8, 0.5, 0.3, 0.2, 0.1, 0.1, 0.1]
    },
    'School': {
        'space_heating': 110, 'dhw': 20, 'color': 'orange', 'simultaneity_factor': 0.75,
        'daily_profile': [0.1, 0.1, 0.1, 0.1, 0.2, 0.5, 0.9, 1.0, 0.9, 0.8, 0.7, 0.7,
                          0.7, 0.7, 0.8, 0.8, 0.7, 0.5, 0.3, 0.2, 0.1, 0.1, 0.1, 0.1]
    },
    'Hospital': {
        'space_heating': 150, 'dhw': 80, 'color': 'red', 'simultaneity_factor': 0.9,
        'daily_profile': [0.7, 0.6, 0.5, 0.5, 0.6, 0.7, 0.8, 0.9, 0.9, 0.8, 0.8, 0.8,
                          0.8, 0.8, 0.8, 0.9, 0.9, 0.9, 0.8, 0.7, 0.7, 0.6, 0.6, 0.6]
    },
    'Industrial': {
        'space_heating': 200, 'dhw': 30, 'color': 'purple', 'simultaneity_factor': 0.85,
        'daily_profile': [0.3, 0.2, 0.1, 0.1, 0.2, 0.5, 0.8, 1.0, 1.0, 1.0, 1.0, 1.0,
                          1.0, 1.0, 1.0, 1.0, 1.0, 0.9, 0.7, 0.5, 0.4, 0.3, 0.3, 0.3]
    }
}

# Pipe material properties
# Each material has: cost_per_m, roughness (mm), thermal_conductivity (W/mK),
#                    temperature limits, insulation_thickness (m)
PIPE_MATERIALS = {
    'Steel': {
        'cost_per_m': 150, 'color': 'red', 'roughness': 0.15,   # mm, nPro default steel library
        'thermal_conductivity': 50,
        'max_temp': 120, 'min_temp': -20,
        'insulation_thickness': 0.01
    },
    'PEX': {
        'cost_per_m': 80, 'color': 'blue', 'roughness': 0.007,
        'thermal_conductivity': 0.4, 'max_temp': 95, 'min_temp': -40,
        'insulation_thickness': 0.02
    },
    'Copper': {
        'cost_per_m': 200, 'color': 'green', 'roughness': 0.0015,
        'thermal_conductivity': 400, 'max_temp': 200, 'min_temp': -100,
        'insulation_thickness': 0.025
    }
}

# System parameters for the district heating network.
# Defines operating temperatures, design constraints, and physical properties.
SYSTEM_PARAMS = {
    'supply_temp': 65,          # °C  (↑ ΔT lowers required flow)
    'return_temp': 45,          # °C  => ΔT = 20 K
    'design_temp_diff': 20,
    'max_velocity': 2.0,        # m/s
    'min_velocity': 0.10,       # m/s  (helps avoid over-sizing)
    'default_material': 'Steel',
    'ground_temp': 10,          # °C
    'operating_hours': 8760,    # h/a — network kept at constant temperatures all year (as in nPro)
    'pump_efficiency': 0.7,
    'water_density': 1000,      # kg/m³
    'water_specific_heat': 4.186,  # kJ/kgK
    'insulation_lambda': 0.027,    # W/mK — nPro default steel pipe library (Appendix F, Table F.1)
    'burial_depth': 1.0,           # m — nPro default mean installation depth; used only if DEPTH_MODE = 'fixed'
    'pair_factor': 2.0,            # supply + return counted as a pair
    'soil_lambda': 1.2             # W/mK — thermal conductivity of soil (moist, DIN EN 13941)
}

# Steel single-pipe library = nPro default library (thesis Appendix F, Table F.1),
# DN15 and DN20 added with the dimensions of Table F.2.
#   inner, outer = steel pipe inner / outer diameter (mm)
#   casing       = outer diameter of the PE casing (mm); casing wall CASING_THICKNESS_MM
#   depth        = installation depth (m)
# Insulation thickness = (casing - 2*casing wall - outer) / 2  (e.g. DN25: 35.2 mm)
# DN15/DN20 are not in the nPro default library: their casing is set so that the insulation
# thickness equals that of DN25 (35.2 mm) and their depth equals DN25 — assumption of this work.
STEEL_PIPE_LIBRARY = {
    #  DN: (inner, outer,  casing, depth)
    15:  (16.1,  21.3,   97.7, 0.8),
    20:  (21.7,  26.9,  103.3, 0.8),
    25:  (27.3,  33.7,  110.0, 0.8),
    32:  (36.0,  42.4,  125.0, 0.8),
    40:  (41.1,  48.3,  125.0, 0.8),
    50:  (53.1,  60.3,  140.0, 0.9),
    65:  (67.1,  76.1,  160.0, 0.9),
    80:  (79.9,  88.9,  180.0, 1.0),
    100: (105.3, 114.3, 200.0, 1.0),
    125: (130.7, 139.7, 225.0, 1.1),
    150: (159.3, 168.3, 250.0, 1.2),
    # DN200 and larger: add the values from the nPro pipe library if they are needed.
}
CASING_THICKNESS_MM = 3.0
# 'per_dn' = depth of the pipe library above; 'fixed' = SYSTEM_PARAMS['burial_depth'] for all pipes.
# Check in nPro which of the two it uses and set this accordingly.
DEPTH_MODE = 'per_dn'
# Smallest DN offered to the sizing algorithm (nPro library with DN15 and DN20 added).
MIN_DN = 15
DN_INNER_MM = {dn: v[0] for dn, v in STEEL_PIPE_LIBRARY.items()}

# Space-heating peak load.
#   'profile': hourly degree-hour profile from the typical meteorological year (TMY) of the
#              location (PVGIS, EU JRC). Peak = annual SH demand / full-load hours, with
#              full-load hours = sum(max(0, T_base - T_out)) / (T_base - min(T_out)).
#   'factor' : previous method, annual / 8760 h x peak_to_average x peak_safety_factor.
# If the weather data cannot be downloaded, the tool falls back to 'factor'.
HEATING_PEAK = {
    'method': 'profile',
    'T_base': 15.0,            # °C heating limit temperature (assumption; nPro: 10–17 °C by insulation standard)
    'safety_factor': 1.0,      # extra factor on the profile peak (1.0 = none)
    'T_design': None,          # °C design outdoor temperature (e.g. from DIN/TS 12831-1 for the site);
                               # None = coldest hour of the TMY. A typical year rarely reaches the design
                               # temperature, so None gives a lower peak than standard heat-load sizing.
}

# Domestic hot water (DHW) peak load per pipe, evaluated per dwelling.
#   Q_DHW(N) = a*N + b*sqrt(N) + c   with  c = 13.1*(q_max/32.3)**2.3,  b = q_max - a - c
#   (Thorsen & Kristjansson, 2006). For N = 1 the result equals q_max.
DHW_PARAMS = {
    'q_max_dwelling_kW': 32.3,     # design load of one dwelling with instantaneous DHW preparation
                                   # (Thorsen & Kristjansson, 2006; 32 kW in Dalla Rosa & Christensen, 2011)
    'a_kW': 1.19,                  # long-term average DHW load per dwelling for large N (Thorsen & Kristjansson, 2006)
    'area_per_dwelling_m2': 94.0,  # average dwelling size in Germany, end of 2025 (Destatis, 2026)
}

# OSM 'building' tags of unheated outbuildings; these are not imported as consumers.
UNHEATED_BUILDING_TAGS = {
    "garage", "garages", "carport", "shed", "roof", "hut", "cabin", "greenhouse",
    "barn", "stable", "cowshed", "sty", "farm_auxiliary", "service", "transformer_tower",
    "parking", "ruins", "construction", "bunker", "toilets", "kiosk", "container",
    "tank", "silo", "bridge", "boathouse", "shelter",
}
# Untagged buildings (building=yes) without a house number and with a footprint below this
# value are treated as outbuildings (garages, sheds). Assumption of this work.
MIN_HEATED_FOOTPRINT_M2 = 40.0

# Space-heating diversity factor (Winter et al., 2001).
# False = phi = 1 (used for the validation against nPro); True = apply DF(n) to space heating.
APPLY_SH_DIVERSITY = False

# Map style constants
PIPE_WEIGHT_PX = 14
PIPE_OPACITY   = 0.95
LABEL_COLOR    = "#111"
LABEL_SIZE_PX  = 12

PIPE_VISUAL_WIDTH_PX = 14

PIPE_STYLE_CHOICES = {
    "Diameter (DN)": ("diameter", lambda v: f"DN {int(round(v))}"),
    "Pressure loss (Pa/m)": ("pressure_loss_pa_per_m", lambda v: f"{v:.0f} Pa/m"),
    "Flow rate (m³/day)": ("flow_rate", lambda v: f"{v:.1f} m³/d"),
    "Heat losses (MWh/a)": ("heat_loss", lambda v: f"{v:.1f} MWh/a"),
    "Temperature (°C)": ("temp_out", lambda v: f"{v:.1f} °C"),
}

# Color stops for the continuous metric colormap (blue → green → yellow → red)
COLOR_STOPS = [
    (0.00, ( 49, 130, 189)),
    (0.33, ( 65, 171,  93)),
    (0.66, (254, 178,  76)),
    (1.00, (240,  59,  32)),
]

# Continuous colormaps for pipe metrics
DP_CMAP   = LinearColormap(colors=['#2c7bb6','#abd9e9','#ffffbf','#fdae61','#d7191c'], vmin=50,  vmax=400)
FLOW_CMAP = LinearColormap(colors=['#edf8fb','#b2e2e2','#66c2a4','#2ca25f','#006d2c'], vmin=0,   vmax=120)
HL_CMAP   = LinearColormap(colors=['#f7fcfd','#ccece6','#66c2a4','#238b45','#00441b'], vmin=0,   vmax=20)

# Colormap used for the DN-based pipe coloring feature (light blue → dark red)
DN_COLORMAP = LinearColormap(
    colors=["#add8e6", "#8b0000"],
    vmin=min(DN_SIZES),
    vmax=max(DN_SIZES),
    caption="Pipe DN (mm)"
)

# Per-DN discrete color lookup (fallback for categorical legend)
DN_COLOR = {
    15:"#6baed6", 20:"#74c476", 25:"#31a354", 32:"#fd8d3c", 40:"#e34a33",
    50:"#b2182b", 65:"#2166ac", 80:"#542788", 100:"#7f0000", 125:"#525252", 150:"#252525"
}



# ============================================================
# SECTION: Caching / OSM Utilities
# ============================================================

# OSMnx Overpass settings (safer mirror + sane timeouts)
ox.settings.use_cache = True
ox.settings.log_console = False
# Overpass settings (work with OSMnx 1.x and 2.x). The endpoint is the base URL WITHOUT
# "/interpreter" — OSMnx appends "/interpreter" and "/status" itself.
OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api",                 # main instance (FOSSGIS)
    "https://overpass.private.coffee/api",         # public instance, no rate limit
    "https://maps.mail.ru/osm/tools/overpass/api", # public instance (VK Maps)
]
OVERPASS_TIMEOUT_S = 90
# Wait times (s) between retry rounds over all servers (exponential backoff, as the
# Overpass servers ask for when they are busy: HTTP 429/502/504).
OVERPASS_BACKOFF_S = [10, 20, 40]
# The public servers reject requests without a meaningful User-Agent (HTTP 406/429).
# Put your own name or e-mail here.
OVERPASS_USER_AGENT = "DHN-Planner/1.0 (THI master thesis; contact: your-email@example.com)"


def _set_overpass(url: str):
    for attr in ("overpass_url", "overpass_endpoint"):      # 2.x / 1.x name
        if hasattr(ox.settings, attr):
            setattr(ox.settings, attr, url)


for _attr, _val in (("requests_timeout", OVERPASS_TIMEOUT_S), ("timeout", OVERPASS_TIMEOUT_S)):
    if hasattr(ox.settings, _attr):
        setattr(ox.settings, _attr, _val)
# No status-page polling before each request (it can add long waits); fair use is kept by
# querying one small area at a time and trying the next server only after a failure.
ox.settings.overpass_rate_limit = False
if hasattr(ox.settings, "http_user_agent"):
    ox.settings.http_user_agent = OVERPASS_USER_AGENT
_set_overpass(OVERPASS_ENDPOINTS[0])


@st.cache_data(show_spinner=False, hash_funcs={
    gpd.GeoDataFrame: lambda gdf: (
        len(gdf),
        tuple(np.round(gdf.total_bounds, 6)),
        int((gdf.geometry.length.fillna(0).sum()) * 1e6)
    )
})
def simplify_for_map(gdf: gpd.GeoDataFrame, tol: float) -> gpd.GeoDataFrame:
    """Return a simplified copy of the GeoDataFrame for faster map rendering.

    Original geometries are left unchanged; only the copy used for display
    is simplified using Shapely's Douglas-Peucker algorithm.

    Args:
        gdf: Input GeoDataFrame (in any CRS).
        tol: Simplification tolerance in the GDF's native units.

    Returns:
        Simplified GeoDataFrame copy.
    """
    if gdf is None or gdf.empty:
        return gdf
    out = gdf.copy()
    out["geometry"] = out.geometry.simplify(tol, preserve_topology=True)
    return out


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_osm_buildings(geojson_geom: dict) -> gpd.GeoDataFrame:
    """Return OSM buildings within a drawn polygon.

    Tries the Overpass servers in OVERPASS_ENDPOINTS one after another, each with a real
    request timeout. Successful results are cached (here for 1 h, and on disk by OSMnx),
    so the same area loads instantly the second time. If every server fails, a
    RuntimeError is raised (failures are not cached), so the user sees that the servers
    were busy instead of "no buildings found".

    Args:
        geojson_geom: GeoJSON geometry dict (Polygon type expected).

    Returns:
        GeoDataFrame with polygonal building footprints, CRS EPSG:4326.
    """
    poly = shape(geojson_geom)
    poly = gpd.GeoSeries([poly], crs="EPSG:4326").iloc[0]

    get_poly = getattr(ox, "features_from_polygon", None) or getattr(ox, "geometries_from_polygon", None)
    if get_poly is None:
        raise RuntimeError("This OSMnx version has no features_from_polygon().")

    import time as _time
    gdf = None
    errors = []
    for _round, _wait in enumerate([0] + list(OVERPASS_BACKOFF_S)):
        if _wait:
            _time.sleep(_wait)                     # servers busy -> back off, then retry
        for ep in OVERPASS_ENDPOINTS:
            _set_overpass(ep)
            try:
                gdf = get_poly(poly, tags={"building": True})
                break                              # success (may be empty = no buildings)
            except Exception as e:
                msg = str(e)
                if "no matching features" in msg.lower() or "InsufficientResponse" in type(e).__name__:
                    gdf = gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs="EPSG:4326")
                    break                          # area really has no buildings
                errors.append(f"{ep.split('/')[2]}: {type(e).__name__}")
        if gdf is not None:
            break
    _set_overpass(OVERPASS_ENDPOINTS[0])

    if gdf is None:
        raise RuntimeError("OSM servers did not respond (" + "; ".join(errors[-3:]) +
                           "). Wait a minute and try again, or draw a smaller area.")

    # Ensure CRS, keep polygonal footprints only
    if gdf.crs is None:
        gdf.set_crs("EPSG:4326", inplace=True)
    else:
        gdf = gdf.to_crs("EPSG:4326")
    gdf = gdf[gdf.geometry.notnull() & gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()

    # Clip to the exact drawn polygon
    try:
        clipper = gpd.GeoDataFrame(geometry=[poly], crs="EPSG:4326")
        gdf = gpd.clip(gdf, clipper)
    except Exception:
        gdf = gdf[gdf.intersects(poly)].copy()

    return gdf


# ============================================================
# SECTION: Physics & Hydraulic Functions
# ============================================================

def reynolds_number(v: float, d: float, nu: float) -> float:
    """Compute the Reynolds number.

    Formula: Re = v * d / nu  (dimensionless)

    Args:
        v:  Flow velocity (m/s).
        d:  Inner pipe diameter (m).
        nu: Kinematic viscosity of water (m²/s).

    Returns:
        Reynolds number, floored at 1.0 to avoid division by zero.
    """
    return max(v * d / max(nu, 1e-12), 1.0)


def friction_factor_haaland(Re: float, roughness: float, d: float) -> float:
    """Compute the Darcy-Weisbach friction factor using the Haaland approximation.

    Regime selection:
      - Laminar  (Re < 2000): λ = 64 / Re   (Hagen-Poiseuille)
      - Turbulent (Re ≥ 2000): Haaland (1983) explicit approximation of
        the Colebrook-White equation.
        λ = 1 / (-1.8 log10[(ε/3.7d)^1.11 + 6.9/Re])²

    Args:
        Re:        Reynolds number (dimensionless).
        roughness: Absolute pipe roughness in mm (as stored in PIPE_MATERIALS).
        d:         Inner pipe diameter in m.

    Returns:
        Darcy friction factor λ (dimensionless).
    """
    if Re < 2000.0:
        return 64.0 / Re

    # Convert roughness from mm to m
    eps = float(roughness) / 1000.0

    return 1.0 / (
        -1.8 * math.log10((eps / (3.7 * d)) ** 1.11 + 6.9 / Re)
    ) ** 2


def pipe_hydraulics(dn_mm: float, q_m3s: float, nu: float,
                    rho: float, roughness: float):
    """Compute hydraulic quantities for a straight pipe segment.

    Uses Darcy-Weisbach: Δp/L = λ * (ρ/2) * v² / d

    Args:
        dn_mm:    Nominal pipe diameter (mm).
        q_m3s:    Volumetric flow rate (m³/s).
        nu:       Kinematic viscosity (m²/s).
        rho:      Fluid density (kg/m³).
        roughness: Pipe roughness (mm), passed to friction_factor_haaland.

    Returns:
        Tuple (d_m, v, Re, f, dp_per_m):
            d_m       – inner diameter (m)
            v         – mean velocity (m/s)
            Re        – Reynolds number
            f         – Darcy friction factor
            dp_per_m  – pressure gradient (Pa/m)
    """
    d = max(DN_INNER_MM.get(int(dn_mm), dn_mm) / 1000.0, 1e-9)   # actual inner diameter
    area = math.pi * d ** 2 / 4.0
    v = q_m3s / max(area, 1e-12)
    Re = reynolds_number(v, d, nu)
    f = friction_factor_haaland(Re, roughness, d)
    dp_per_m = f * 0.5 * rho * v ** 2 / max(d, 1e-12)
    return d, v, Re, f, dp_per_m


def pipe_radial_resistances(d_m: float, material: str,
                             lambda_ins: float, lambda_soil: float,
                             burial: float, dn: int = None):
    """Compute radial thermal resistances for a buried insulated pipe.

    Based on DIN EN 13941 / Eq. (3.15)-(3.16) of the thesis. For steel pipes listed in
    STEEL_PIPE_LIBRARY the insulation lies between the steel outer diameter and the inner
    side of the casing, and the soil resistance uses the casing outer radius and the
    installation depth of that DN (if DEPTH_MODE = 'per_dn').

    Returns:
        Tuple (R_ins, R_soil, r_i) in m·K/W per m length and m.
    """
    r_i = max(d_m / 2.0, 1e-9)
    lib = STEEL_PIPE_LIBRARY.get(int(dn)) if (dn is not None and material == 'Steel') else None
    if lib:
        _inner, outer, casing, depth = lib
        r_steel = outer / 2000.0
        r_ins_outer = (casing - 2.0 * CASING_THICKNESS_MM) / 2000.0
        r_casing = casing / 2000.0
        h = depth if DEPTH_MODE == 'per_dn' else burial
    else:
        ins_thk = float(PIPE_MATERIALS.get(material, PIPE_MATERIALS['Steel'])['insulation_thickness'])
        r_steel = r_i
        r_ins_outer = r_i + ins_thk
        r_casing = r_ins_outer
        h = burial

    R_ins = math.log(max(r_ins_outer / r_steel, 1.000001)) / (2.0 * math.pi * max(lambda_ins, 1e-9))
    R_soil = math.log(max(2.0 * h / max(r_casing, 1e-9), 1.000001)) / (
        2.0 * math.pi * max(lambda_soil, 1e-9)
    )
    return R_ins, R_soil, r_i


def pipe_heat_loss_and_temp(T_avg, T_soil, length_m, hours,
                             d_m, material, rho, q_m3s,
                             lambda_ins, lambda_soil, burial, pair_factor,
                             pipe_type, cp_kJ, T_in_override=None, dn=None):
    """Compute annual heat loss and outlet temperature for a pipe segment.

    Implements Eq. (3.2) + (3.3) from the thesis combined with an
    exponential T(x) temperature decay model:
        T(x) = T_soil + (T_in - T_soil) * exp(-K * x)
    where K = k_per_m / (m_dot * cp).

    Sequential temperature propagation: T_in_override carries the actual
    inlet temperature from the upstream pipe's outlet (nPro-consistent).
    If None, falls back to the global supply temperature.

    Args:
        T_avg:          Average of supply and return temperatures (°C).
        T_soil:         Ground temperature (°C).
        length_m:       Pipe segment length (m).
        hours:          Annual operating hours (h/a).
        d_m:            Inner diameter (m).
        material:       Pipe material key.
        rho:            Water density (kg/m³).
        q_m3s:          Volumetric flow rate (m³/s).
        lambda_ins:     Insulation thermal conductivity (W/mK).
        lambda_soil:    Soil thermal conductivity (W/mK).
        burial:         Burial depth (m).
        pair_factor:    1.0 for single pipe, 2.0 for supply+return pair.
        pipe_type:      'main' or 'connection' — pair_factor applied for main.
        cp_kJ:          Specific heat capacity (kJ/kg·K).
        T_in_override:  Inlet temperature override (°C). None → use supply_temp.

    Returns:
        Tuple (annual_heat_loss_MWh, T_out, dT_seg).
    """
    R_ins, R_soil, _ = pipe_radial_resistances(d_m, material, lambda_ins, lambda_soil, burial, dn=dn)
    q_Wpm = (T_avg - T_soil) / (R_ins + R_soil)
    if pipe_type in ('main', 'connection'):
        q_Wpm *= pair_factor

    annual_heat_loss_MWh = (q_Wpm * length_m * hours) / 1e6

    k_per_m_single = 1.0 / (R_ins + R_soil)
    m_dot = rho * q_m3s
    cp_J = cp_kJ * 1000.0

    if m_dot * cp_J <= 1e-12:
        T_out = T_avg
    else:
        T_in = T_in_override if T_in_override is not None else float(SYSTEM_PARAMS['supply_temp'])
        K = k_per_m_single / (m_dot * cp_J)
        T_out = T_soil + (T_in - T_soil) * math.exp(-K * length_m)

    dT_seg = float(float(SYSTEM_PARAMS['supply_temp']) - T_out)
    return float(annual_heat_loss_MWh), float(T_out), dT_seg


# ============================================================
# SECTION: Map Rendering Helpers
# ============================================================

def _interp(a, b, t):
    """Linearly interpolate between two RGB tuples at parameter t in [0, 1]."""
    return tuple(int(round(a[i] + (b[i] - a[i]) * t)) for i in range(3))


def metric_to_color(val: float, vmin: float, vmax: float) -> str:
    """Map a scalar value to a hex color using the piecewise COLOR_STOPS ramp.

    The ramp goes blue → green → yellow → red over the [vmin, vmax] range.

    Args:
        val:  The value to color.
        vmin: Minimum of the range (maps to blue).
        vmax: Maximum of the range (maps to red).

    Returns:
        Hex color string, e.g. '#3182bd'.
    """
    if vmin is None or vmax is None or vmax <= vmin:
        r, g, b = 49, 130, 189
        return f"#{r:02x}{g:02x}{b:02x}"
    t = (val - vmin) / (vmax - vmin)
    t = max(0.0, min(1.0, t))
    for (x0, c0), (x1, c1) in zip(COLOR_STOPS[:-1], COLOR_STOPS[1:]):
        if t <= x1:
            lt = (t - x0) / max((x1 - x0), 1e-9)
            r, g, b = _interp(c0, c1, lt)
            return f"#{r:02x}{g:02x}{b:02x}"
    r, g, b = COLOR_STOPS[-1][1]
    return f"#{r:02x}{g:02x}{b:02x}"


def _nearest_dn_color(dn: float) -> str:
    """Return the discrete hex color for the nearest DN size in DN_COLOR."""
    if not dn:
        return "#3388ff"
    key = min(DN_COLOR.keys(), key=lambda k: abs(k - dn))
    return DN_COLOR[key]


def _pipe_color(pipe: dict, mode: str) -> str:
    """Compute the display color of a pipe based on the selected visualization mode.

    Uses discrete color brackets consistent with the map legend.

    Args:
        pipe: Pipe dict with a 'parameters' sub-dict.
        mode: One of the PIPE_STYLE_CHOICES keys.

    Returns:
        Hex color string.
    """
    p = pipe.get("parameters", {})
    if mode == "Diameter (DN)":
        return _discrete_color(float(p.get("diameter", 0) or 0), "Diameter (DN)")
    if mode == "Pressure loss (Pa/m)":
        return _discrete_color(float(p.get("pressure_loss_pa_per_m", 0) or 0), "Pressure loss (Pa/m)")
    if mode in ("Heat losses (MWh/a)", "Heat loss (MWh/a)"):
        return _discrete_color(float(p.get("heat_loss", 0) or 0), "Heat losses (MWh/a)")
    if mode == "Flow rate (m³/day)":
        return _discrete_color(float(p.get("flow_rate", 0) or 0), "Flow rate (m³/day)")
    if mode == "Temperature (°C)":
        return _discrete_color(float(p.get("temp_out", 0) or 0), "Temperature (°C)")
    return "#cf3d3d" if pipe.get("type") == "main" else "#2b8cbe"


def _pipe_weight(pipe: dict, mode: str) -> int:
    """Compute display line weight (pixels) for a pipe.

    Args:
        pipe: Pipe dict with 'parameters' sub-dict.
        mode: Visualization mode string.

    Returns:
        Integer pixel width.
    """
    p = pipe.get("parameters", {})
    if mode == "Diameter (DN)":
        dn = float(p.get("diameter", 0) or 0)
        return max(2, min(12, int(round(dn / 10))))
    if mode == "Flow rate (m³/day)":
        q = float(p.get("flow_rate", 0) or 0)
        return max(2, min(14, int((q ** 0.5))))
    return 4 if pipe.get("type") == "main" else 3


# Discrete color brackets for each pipe metric
# Unified color scale: green (low/good) → yellow → orange → red (high/critical)
# Same 6-step palette across all metrics for visual consistency
# Color scale: blue (best/lowest) → dark green → light green → yellow → orange → red (worst/highest)
_C1, _C2, _C3, _C4, _C5, _C6 = "#08519c", "#3182bd", "#6baed6", "#fed976", "#fd8d3c", "#800026"
_DN_BRACKETS = [
    (20,   _C1, "DN ≤ 20"),
    (32,   _C2, "DN ≤ 32"),
    (50,   _C3, "DN ≤ 50"),
    (80,   _C4, "DN ≤ 80"),
    (125,  _C5, "DN ≤ 125"),
    (None, _C6, "DN > 125"),
]
_DP_BRACKETS = [
    (55,   _C1, "≤ 55 Pa/m"),
    (110,  _C2, "≤ 110 Pa/m"),
    (165,  _C3, "≤ 165 Pa/m"),
    (220,  _C4, "≤ 220 Pa/m"),
    (276,  _C5, "≤ 276 Pa/m"),
    (None, _C6, "> 276 Pa/m"),
]
_FLOW_BRACKETS = [
    (5,    _C1, "≤ 5 m³/day"),
    (20,   _C2, "≤ 20 m³/day"),
    (50,   _C3, "≤ 50 m³/day"),
    (100,  _C4, "≤ 100 m³/day"),
    (200,  _C5, "≤ 200 m³/day"),
    (None, _C6, "> 200 m³/day"),
]
_HL_BRACKETS = [
    (1,    _C1, "≤ 1 MWh/a"),
    (5,    _C2, "≤ 5 MWh/a"),
    (10,   _C3, "≤ 10 MWh/a"),
    (20,   _C4, "≤ 20 MWh/a"),
    (50,   _C5, "≤ 50 MWh/a"),
    (None, _C6, "> 50 MWh/a"),
]
# Temperature: high temp = good (green), low temp = bad (red) — reversed scale
_TEMP_BRACKETS = [
    (64.5, _C1, "≥ 64.5 °C"),
    (64.0, _C2, "≥ 64.0 °C"),
    (63.5, _C3, "≥ 63.5 °C"),
    (63.0, _C4, "≥ 63.0 °C"),
    (62.0, _C5, "≥ 62.0 °C"),
    (None, _C6, "< 62.0 °C"),
]


def _discrete_color(val: float, metric_choice: str) -> str:
    """Return the discrete hex color for a pipe metric value.

    Args:
        val:           The metric value.
        metric_choice: One of the PIPE_STYLE_CHOICES keys.

    Returns:
        Hex color string.
    """
    if metric_choice == "Diameter (DN)":
        for threshold, color, _ in _DN_BRACKETS:
            if threshold is None or val <= threshold:
                return color
    elif metric_choice == "Pressure loss (Pa/m)":
        for threshold, color, _ in _DP_BRACKETS:
            if threshold is None or val <= threshold:
                return color
    elif metric_choice == "Flow rate (m³/day)":
        for threshold, color, _ in _FLOW_BRACKETS:
            if threshold is None or val <= threshold:
                return color
    elif metric_choice in ("Heat losses (MWh/a)", "Heat loss (MWh/a)"):
        for threshold, color, _ in _HL_BRACKETS:
            if threshold is None or val <= threshold:
                return color
    elif metric_choice == "Temperature (°C)":
        for threshold, color, _ in _TEMP_BRACKETS:
            if threshold is None or val >= threshold:
                return color
        return _TEMP_BRACKETS[-1][1]
    # Fallback
    return "#cf3d3d"


def _make_discrete_legend(metric_choice: str, pipes: list):
    """Return a Folium MacroElement that renders a legend overlay inside the map iframe.

    The legend is positioned in the bottom-right corner of the map using a
    Leaflet custom control, which keeps it properly inside the map container.

    Args:
        metric_choice: The currently selected pipe visualization metric.
        pipes:         Unused; kept for API compatibility.

    Returns:
        MacroElement to add_to(m), or None if metric has no discrete brackets.
    """
    bracket_map = {
        "Diameter (DN)":          (_DN_BRACKETS,   "Pipe DN"),
        "Pressure loss (Pa/m)":   (_DP_BRACKETS,   "Pressure loss"),
        "Flow rate (m³/day)":     (_FLOW_BRACKETS, "Flow rate"),
        "Heat losses (MWh/a)":    (_HL_BRACKETS,   "Heat losses"),
        "Heat loss (MWh/a)":      (_HL_BRACKETS,   "Heat losses"),
        "Temperature (°C)":       (_TEMP_BRACKETS, "Temperature"),
    }
    if metric_choice not in bracket_map:
        return None

    brackets, title = bracket_map[metric_choice]

    # Build the legend rows HTML
    rows_html = "".join(
        f'<div style="display:flex;align-items:center;margin-bottom:3px;">'
        f'<span style="display:inline-block;width:16px;height:16px;background:{color};'
        f'border:1px solid #666;margin-right:6px;flex-shrink:0;border-radius:2px;"></span>'
        f'<span style="font-size:11px;color:#222;">{label}</span></div>'
        for _, color, label in brackets
    )

    # Use folium.Element to inject a <style>+<div> directly into the map HTML.
    # This avoids Jinja2 Template entirely (which chokes on % in CSS strings).
    inner = f'<div class="dhn-legend-title">{title}</div>' + rows_html
    css = ('<style>.dhn-legend{background:rgba(255,255,255,0.93);padding:8px 12px;border-radius:6px;'
           'border:1px solid #aaa;box-shadow:0 2px 6px rgba(0,0,0,0.2);font-family:Arial,sans-serif;'
           'min-width:130px;}'
           '.dhn-legend-title{font-size:11px;font-weight:700;color:#222;margin-bottom:5px;'
           'border-bottom:1px solid #ddd;padding-bottom:3px;}</style>')
    macro = MacroElement()
    macro._template = Template(
        "{% macro header(this, kwargs) %}" + css + "{% endmacro %}"
                                                   "{% macro script(this, kwargs) %}"
                                                   "var dhnLegend = L.control({position: 'topright'});"
                                                   "dhnLegend.onAdd = function(map) {"
                                                   "  var div = L.DomUtil.create('div', 'dhn-legend');"
                                                   "  div.innerHTML = " + json.dumps(inner) + ";"
                                                                                              "  return div;"
                                                                                              "};"
                                                                                              "dhnLegend.addTo({{ this._parent.get_name() }});"
                                                                                              "{% endmacro %}")
    return macro


def _legend_for(color_by: str) -> str:
    """Kept for backward compatibility; delegates to _make_discrete_legend."""
    return _make_discrete_legend(color_by, [])


def _coords_latlon(geom) -> list:
    """Convert a pipe geometry to Folium-ready [(lat, lon), ...] pairs.

    Args:
        geom: GeoJSON geometry dict or Shapely geometry.

    Returns:
        List of (lat, lon) tuples.
    """
    line = shape(geom)
    return [(y, x) for (x, y) in line.coords]



# ============================================================
# SECTION: Heat Demand Functions
# ============================================================

_TMY_CACHE = {}


def get_hourly_temperature(lat: float, lon: float):
    """Hourly 2 m air temperature (°C, 8760 values) of the typical meteorological year
    at (lat, lon) from PVGIS (EU Joint Research Centre). Returns None if unavailable."""
    key = (round(float(lat), 2), round(float(lon), 2))
    if key in _TMY_CACHE:
        return _TMY_CACHE[key]
    import requests
    temps, errors = None, []
    for url in ("https://re.jrc.ec.europa.eu/api/v5_3/tmy",
                "https://re.jrc.ec.europa.eu/api/v5_2/tmy",
                "https://re.jrc.ec.europa.eu/api/tmy"):
        try:
            r = requests.get(url, params={'lat': key[0], 'lon': key[1], 'outputformat': 'json'}, timeout=60)
            r.raise_for_status()
            rows = r.json()['outputs']['tmy_hourly']
            temps = [float(x['T2m']) for x in rows]
            if len(temps) >= 8760:
                break
            temps = None
        except Exception as e:
            errors.append(f"{url}: {type(e).__name__}: {str(e)[:150]}")
            temps = None
    if temps:
        _TMY_CACHE[key] = temps
    else:
        st.warning("PVGIS weather download failed for "
                   f"{key[0]:.2f} N, {key[1]:.2f} E:\n\n" + "\n\n".join(errors))
    return temps


def heating_full_load_hours(lat: float, lon: float, t_base: float):
    """Degree-hour profile of the location: returns dict with full-load hours, the minimum
    outdoor temperature and the normalised hourly profile (sums to 1), or None."""
    temps = get_hourly_temperature(lat, lon)
    if not temps:
        return None
    dh = [max(0.0, t_base - t) for t in temps]
    total = sum(dh)
    t_min = min(temps)
    t_des = HEATING_PEAK.get('T_design')
    if t_des is not None:
        t_min = min(t_min, float(t_des))
    if total <= 0 or t_base <= t_min:
        return None
    return {
        'flh': total / (t_base - t_min),       # h/a: annual demand / peak
        't_min': t_min,
        't_base': t_base,
        'degree_hours': total,
        'profile': [x / total for x in dh],    # share of annual SH demand per hour
        'location': key_str(lat, lon),
    }


def key_str(lat: float, lon: float) -> str:
    return f"{lat:.2f} N, {lon:.2f} E"


def dhw_peak_kW(n_dwellings: int, q_max: float = None) -> float:
    """Simultaneous DHW peak load (kW) of n dwellings (Thorsen & Kristjansson, 2006)."""
    if n_dwellings <= 0:
        return 0.0
    q_max = float(DHW_PARAMS['q_max_dwelling_kW'] if q_max is None else q_max)
    a = float(DHW_PARAMS['a_kW'])
    c = 13.1 * (q_max / 32.3) ** 2.3
    b = q_max - a - c
    return a * n_dwellings + b * math.sqrt(n_dwellings) + c

# Heat transfer station of one building: central DHW preparation with buffer storage and
# priority for DHW (sizing method of the nPro documentation; inputs = nPro default values).
DHW_STATION = {
    'tap_flow_l_min':   16.0,   # flow rate per outlet ("Standard", approx. 35 kW)
    'T_tap':            42.0,   # °C mixed water at the outlet (shower)
    'T_cold':           10.0,   # °C cold water
    'storage_spread_K': 30.0,   # K storage temperature spread
    'draw_off_min':     10.0,   # min draw-off duration
    'recharge_min':     30.0,   # min recharge duration
    'storage_eff':      0.85,   # storage efficiency
    'priority':         True,   # True: station power = max(SH peak, DHW charging power)
}


def _standard_storage_l(v_l: float) -> float:
    """Round a storage volume up to standard sizes (min. 200 l; steps 100 l to 1000 l,
    200 l to 2000 l, 500 l to 5000 l, 1000 l above)."""
    if v_l <= 200.0:
        return 200.0
    for step, upto in ((100.0, 1000.0), (200.0, 2000.0), (500.0, 5000.0)):
        if v_l <= upto:
            return math.ceil(v_l / step) * step
    return math.ceil(v_l / 1000.0) * 1000.0


def station_power_kW(sh_peak_kW: float, n_dwellings: int) -> float:
    """Design power (kW) of the heat transfer station of one residential building."""
    p = DHW_STATION
    q_tap = 4.186 * (p['tap_flow_l_min'] / 60.0) * (p['T_tap'] - p['T_cold'])     # kW, one full tap
    q_dhw = dhw_peak_kW(n_dwellings, q_max=q_tap)                                # kW, simultaneous DHW
    v_draw_l = q_dhw / (4.186 * p['storage_spread_K']) * p['draw_off_min'] * 60.0   # l
    v_store_l = _standard_storage_l(v_draw_l / p['storage_eff'])                 # l
    q_charge = 4.186 * (v_store_l / (p['recharge_min'] * 60.0)) * p['storage_spread_K']  # kW
    return max(sh_peak_kW, q_charge) if p['priority'] else sh_peak_kW + q_charge

def n_dwellings_of(floor_area_m2: float) -> int:
    """Number of dwellings of a residential building, estimated from its floor area."""
    return max(1, int(round(float(floor_area_m2 or 0.0) / float(DHW_PARAMS['area_per_dwelling_m2']))))


def calculate_heat_demand(area: float, building_type: str) -> dict:
    """Calculate annual heat demand for a building using DEMAND_PROFILES.

    Args:
        area:          Building floor area (m²).
        building_type: Must be a key in DEMAND_PROFILES.

    Returns:
        Dict with keys: space_heating, dhw, total, peak_demand, color.

    Raises:
        ValueError: If area <= 0 or computed demand is non-positive.
    """
    if area <= 0:
        raise ValueError("Building area must be positive")

    profile = DEMAND_PROFILES.get(building_type, DEMAND_PROFILES['Residential'])

    space_heating = (profile['space_heating'] * area) / 1000  # MWh
    dhw = (profile['dhw'] * area) / 1000                       # MWh

    if space_heating <= 0 or dhw <= 0:
        raise ValueError("Calculated demand must be positive")

    total = space_heating + dhw
    avg_kW = (total * 1000.0) / 8760.0
    peak_factor = profile.get('peak_to_average', 2.5)
    peak_demand = avg_kW * peak_factor  # kW

    return {
        'space_heating': space_heating,
        'dhw': dhw,
        'total': total,
        'peak_demand': peak_demand,
        'color': profile['color']
    }


def recompute_demands_for_all(gdf: gpd.GeoDataFrame):
    """Recompute space_heating, domestic_hot_water, and total_demand for every building.

    Mutates the GeoDataFrame in-place using DEMAND_PROFILES.

    Args:
        gdf: Buildings GeoDataFrame; must have 'building_type' and 'floor_area' columns.
    """
    if gdf is None or gdf.empty:
        return
    for idx, row in gdf.iterrows():
        prof = DEMAND_PROFILES.get(row.get("building_type", "Residential"), DEMAND_PROFILES["Residential"])
        fa = float(row.get("floor_area", 0.0) or 0.0)
        space = (prof["space_heating"] * fa) / 1000.0
        dhw   = (prof["dhw"]           * fa) / 1000.0
        gdf.at[idx, "space_heating"]       = space
        gdf.at[idx, "domestic_hot_water"]  = dhw
        gdf.at[idx, "total_demand"]        = space + dhw


def calculate_building_area(geom) -> float:
    """Return useful floor area (m²) = footprint × assumed_levels × utilization_factor.

    Uses a local UTM projection for accurate area calculation.

    Args:
        geom: Shapely geometry of the building footprint (in EPSG:4326).

    Returns:
        Estimated useful floor area in m².
    """
    try:
        cent = geom.centroid
        zone = int((cent.x + 180) // 6) + 1
        epsg_zone = 32600 + zone if cent.y >= 0 else 32700 + zone
        footprint_m2 = gpd.GeoSeries([geom], crs="EPSG:4326").to_crs(epsg=epsg_zone).area.iloc[0]
        assumed_levels = 3.0
        useful_factor = 0.85
        return float(footprint_m2 * assumed_levels * useful_factor)
    except Exception:
        return float(geom.area * 111319.9 ** 2 * 2.0 * 0.7)


def get_address(row) -> str:
    """Get a formatted address from OSM data or fall back to reverse geocoding.

    Args:
        row: A GeoDataFrame row (Series) or dict with OSM tag columns.

    Returns:
        Address string, or 'Unnamed Building' if not resolvable.
    """
    if isinstance(row, dict) and 'address' in row and row['address'].startswith('Manually Added Building'):
        return row['address']

    street = row.get("addr:street", "")
    housenumber = row.get("addr:housenumber", "")
    if street and housenumber:
        return f"{street} {housenumber}"

    try:
        centroid = row.geometry.centroid
        location = geolocator.reverse((centroid.y, centroid.x), timeout=10)
        address = location.raw.get('address', {})
        road = address.get('road', '')
        house_number = address.get('house_number', '')
        return f"{road} {house_number}".strip() if road else "Unnamed Building"
    except Exception:
        return "Unnamed Building"


import numpy as _np  # local alias used by _is_missing_name


def _is_missing_name(val) -> bool:
    """Return True if a value should be considered a missing/placeholder address."""
    if val is None:
        return True
    try:
        if isinstance(val, float) and _np.isnan(val):
            return True
    except Exception:
        pass
    if not isinstance(val, str):
        return False
    s = val.strip()
    if s == "":
        return True
    if s.lower() in {"nan", "n/a", "none", "unnamed building", "osm building", "nan nan"}:
        return True
    if re.fullmatch(r"[-_\s]+", s):
        return True
    return False


def assign_missing_addresses(gdf: gpd.GeoDataFrame, prefix: str = "Building") -> gpd.GeoDataFrame:
    """Fill missing 'address' values with unique names like 'Building 1', 'Building 2', ...

    Args:
        gdf:    Buildings GeoDataFrame.
        prefix: Prefix for auto-generated names.

    Returns:
        Copy of the GeoDataFrame with 'address' column fully populated.
    """
    if gdf is None or getattr(gdf, "empty", False):
        return gdf

    if "address" not in gdf.columns:
        gdf["address"] = [None] * len(gdf)

    existing = set()
    for v in gdf["address"].astype(object).tolist():
        if v is None:
            continue
        if not _is_missing_name(v):
            existing.add(str(v).strip())

    try:
        other = st.session_state.get("b_gdf")
        if other is not None and len(other) > 0:
            for v in other.get("address", []).astype(object).tolist():
                if v is None:
                    continue
                if not _is_missing_name(v):
                    existing.add(str(v).strip())
    except Exception:
        pass

    max_n = 0
    for name in existing:
        m = re.match(rf"^{re.escape(prefix)}\s*(\d+)$", name, re.IGNORECASE)
        if m:
            try:
                n = int(m.group(1))
                if n > max_n:
                    max_n = n
            except Exception:
                pass

    out_names = []
    counter = max_n + 1
    for val in gdf["address"].astype(object).tolist():
        if _is_missing_name(val):
            while f"{prefix} {counter}" in existing:
                counter += 1
            new_name = f"{prefix} {counter}"
            existing.add(new_name)
            out_names.append(new_name)
            counter += 1
        else:
            out_names.append(str(val).strip())

    gdf = gdf.copy()
    gdf["address"] = out_names
    return gdf


def generate_report(buildings, pipes) -> BytesIO:
    """Generate an Excel report with building and pipe network data.

    Args:
        buildings: Buildings GeoDataFrame.
        pipes:     List of pipe dicts from PipeNetwork.

    Returns:
        BytesIO buffer containing the .xlsx file.
    """
    buffer = BytesIO()
    with pd.ExcelWriter(buffer) as writer:
        buildings.to_excel(writer, sheet_name='Buildings')
        pd.DataFrame(pipes).to_excel(writer, sheet_name='Pipes')
    return buffer



# ============================================================
# SECTION: Network / Graph Utilities
# ============================================================

def _run_flows_async():
    """Thread target: run calculate_network_flows() and update session state flags."""
    try:
        st.session_state.pipe_network.calculate_network_flows()
        st.session_state.flow_done = True
        st.session_state.flow_running = False
    except Exception as e:
        st.session_state.flow_error = str(e)
        st.session_state.flow_running = False


def start_flow_calc_async():
    """Start a background thread for network flow calculation.

    Checks the 'flow_running' flag to prevent concurrent runs.
    Sets flow_done=False and flow_running=True before spawning the thread.
    """
    if st.session_state.get("flow_running"):
        return
    st.session_state.flow_done = False
    st.session_state.flow_error = None
    st.session_state.flow_running = True
    threading.Thread(target=_run_flows_async, daemon=True).start()


def create_star_topology(buildings_gdf: gpd.GeoDataFrame, hub_location: tuple) -> 'PipeNetwork':
    """Create a star topology network connecting all buildings directly to the energy hub.

    Args:
        buildings_gdf: GeoDataFrame of buildings with 'total_demand' column.
        hub_location:  (lat, lon) tuple for the energy hub position.

    Returns:
        Populated PipeNetwork instance with one connection pipe per building.
    """
    network = PipeNetwork()
    hub_node = network.add_node('source', hub_location, 0)

    for idx, building in buildings_gdf.iterrows():
        centroid = building.geometry.centroid
        building_node = network.add_node(
            'building',
            (centroid.y, centroid.x),
            building['total_demand'],
            idx
        )
        line = LineString([(hub_location[1], hub_location[0]), (centroid.x, centroid.y)])
        network.add_polyline_pipe(
            start_node=hub_node,
            end_node=building_node,
            geometry=mapping(line),
            material=SYSTEM_PARAMS['default_material'],
            pipe_type='connection'
        )

    return network


# ============================================================
# SECTION: PipeNetwork Class
# ============================================================

class PipeNetwork:
    """Directed-graph model of a district heating pipe network.

    Nodes represent buildings, junctions, or energy sources.
    Pipes are directed edges carrying hydraulic and thermal parameters.
    """

    def __init__(self):
        """Initialize an empty network with no nodes, pipes, or energy hub."""
        self.nodes = {}          # node_id -> {type, location, demand, building_id}
        self.pipes = []          # list of pipe dicts
        self.graph = DiGraph()   # directed graph for connectivity analysis
        self.next_node_id = 0
        self.next_pipe_id = 1
        self.energy_hub = None       # ID of the designated source node
        self.energy_hub_node = None  # alias kept for compatibility

    def prune_disconnected_pipes(self):
        """Remove pipes that have no path back to the energy hub."""
        from networkx import has_path
        connected_pipes = []
        for pipe in self.pipes:
            if (pipe['start_node'] in self.graph.nodes and
                    pipe['end_node'] in self.graph.nodes and
                    has_path(self.graph, self.energy_hub, pipe['end_node'])):
                connected_pipes.append(pipe)
            else:
                if self.graph.has_edge(pipe['start_node'], pipe['end_node']):
                    self.graph.remove_edge(pipe['start_node'], pipe['end_node'])
        self.pipes = connected_pipes

    def prune_zero_flow_pipes(self, flow_eps_m3day: float = 0.01,
                               heat_eps_MWh: float = 1e-3) -> list:
        """Remove pipes with effectively zero flow or delivered energy.

        Args:
            flow_eps_m3day: Daily flow threshold below which a pipe is removed (m³/day).
            heat_eps_MWh:   Delivered heat threshold below which a pipe is removed (MWh/a).

        Returns:
            List of removed pipe IDs.
        """
        keep = []
        removed_ids = []
        for p in self.pipes:
            params = p.get('parameters', {})
            q_day = float(params.get('flow_rate', 0.0))
            H_MWh = float(params.get('delivered_heat', 0.0))
            if (abs(q_day) <= flow_eps_m3day) or (abs(H_MWh) <= heat_eps_MWh):
                if self.graph.has_edge(p['start_node'], p['end_node']):
                    self.graph.remove_edge(p['start_node'], p['end_node'])
                removed_ids.append(p['id'])
            else:
                keep.append(p)
        self.pipes = keep
        return removed_ids

    def prune_isolated_junctions(self) -> list:
        """Remove junction nodes with zero incident edges after pipe pruning.

        Returns:
            List of removed node IDs.
        """
        to_remove = []
        for nid, ndata in list(self.nodes.items()):
            if ndata.get('type') == 'junction':
                deg = 0
                if self.graph.has_node(nid):
                    deg = self.graph.in_degree(nid) + self.graph.out_degree(nid)
                if deg == 0:
                    to_remove.append(nid)
        for nid in to_remove:
            self.nodes.pop(nid, None)
            if self.graph.has_node(nid):
                self.graph.remove_node(nid)
        return to_remove

    def add_node(self, node_type: str, location: tuple,
                 demand: float = 0, building_id=None) -> str:
        """Add a node to the network.

        Args:
            node_type:   'source', 'building', or 'junction'.
            location:    (latitude, longitude) tuple.
            demand:      Annual heat demand in MWh/a (buildings only).
            building_id: Index into the buildings GeoDataFrame.

        Returns:
            The unique node ID string (e.g. 'building_3').
        """
        node_id = f"{node_type}_{self.next_node_id}"
        self.nodes[node_id] = {
            'type': node_type,
            'location': location,
            'demand': demand,
            'building_id': building_id
        }
        self.graph.add_node(node_id)
        self.next_node_id += 1
        if node_type == 'source' and self.energy_hub is None:
            self.energy_hub = node_id
        return node_id

    def add_pipe_segment(self, start_node: str, end_node: str,
                         geometry: dict, material: str, pipe_type: str) -> int:
        """Add a straight pipe segment between two existing nodes.

        Pipe length is computed using a local UTM projection for accuracy.

        Args:
            start_node: ID of the starting node.
            end_node:   ID of the ending node.
            geometry:   GeoJSON dict (LineString).
            material:   Pipe material key (must be in PIPE_MATERIALS).
            pipe_type:  'main' or 'connection'.

        Returns:
            Integer pipe ID.

        Raises:
            ValueError: If nodes don't exist or the pipe has zero length.
        """
        if start_node not in self.nodes or end_node not in self.nodes:
            raise ValueError("Start or end node does not exist")

        line = shape(geometry)
        if line.length == 0:
            raise ValueError("Pipe length cannot be zero")

        cent = line.centroid
        zone = int((cent.x + 180) // 6) + 1
        epsg_len = (32600 if cent.y >= 0 else 32700) + zone
        length_m = float(gpd.GeoSeries([line], crs="EPSG:4326").to_crs(epsg=epsg_len).length.iloc[0])

        if length_m <= 0:
            raise ValueError("Pipe length cannot be zero")

        pipe_data = {
            'id': self.next_pipe_id,
            'start_node': start_node,
            'end_node': end_node,
            'geometry': geometry,
            'length_m': float(length_m),
            'material': material,
            'type': pipe_type,
            'parameters': {
                'diameter': 0,
                'velocity': 0,
                'pressure_loss': 0,
                'heat_loss': 0,
                'delivered_heat': 0,
                'flow_rate': 0
            }
        }

        self.pipes.append(pipe_data)
        self.graph.add_edge(start_node, end_node, pipe_id=self.next_pipe_id)
        self.next_pipe_id += 1
        return pipe_data['id']

    from shapely.geometry import LineString, shape, mapping

    def add_polyline_pipe(self, start_node: str, end_node: str,
                          geometry: dict, material: str, pipe_type: str):
        """Add a multi-vertex polyline as a series of straight pipe segments.

        Creates junction nodes at each intermediate vertex and calls
        add_pipe_segment() for each straight section.

        Args:
            start_node: ID of the first node.
            end_node:   ID of the last node.
            geometry:   GeoJSON dict (LineString, may have many vertices).
            material:   Pipe material key.
            pipe_type:  'main' or 'connection'.
        """
        line = shape(geometry)
        coords = list(line.coords)

        if len(coords) < 2:
            raise ValueError("Polyline must have at least 2 points")

        prev_node = start_node
        prev_xy = coords[0]

        for xy in coords[1:-1]:
            junc_id = self.add_node('junction', (xy[1], xy[0]))
            seg_geom = LineString([prev_xy, xy])
            self.add_pipe_segment(prev_node, junc_id, mapping(seg_geom), material, pipe_type)
            prev_node = junc_id
            prev_xy = xy

        last_xy = coords[-1]
        seg_geom = LineString([prev_xy, last_xy])
        self.add_pipe_segment(prev_node, end_node, mapping(seg_geom), material, pipe_type)

    def validate_system(self) -> list:
        """Check network validity before hydraulic calculations.

        Returns:
            List of error message strings; empty list means the network is valid.
        """
        errors = []
        if SYSTEM_PARAMS['design_temp_diff'] <= 0:
            errors.append("Temperature difference must be positive")
        building_nodes = [n for n, data in self.nodes.items() if data['type'] == 'building']
        total_demand = sum(self.nodes[n]['demand'] for n in building_nodes)
        if total_demand <= 0:
            errors.append("No buildings with positive demand")
        if not any(data['type'] == 'source' for data in self.nodes.values()):
            errors.append("No energy hub designated")
        if not self.pipes:
            errors.append("No pipes in the network")
        return errors

    def calculate_network_flows(self):
        """Run the mass/energy balance from hub to buildings.

        Algorithm:
          1. For each building, trace the weak shortest path from the hub
             and accumulate peak thermal load (kW) and annual energy (MWh/a)
             on every pipe along the path.
          2. Apply the Winter et al. (2001) diversity factor:
             DF(n) = A + B / (1 + (n/C)^D)  [Euroheat & Power 09&10/2001, Eq. 6-1]
          3. For each pipe, select the smallest DN that satisfies:
             - velocity ≤ v_max
             - pressure gradient in the recommended band [30, 150] Pa/m  (nPro-style)
             Fallback: enforce only v_max; ultimate fallback: largest DN.
          4. Compute heat losses and outlet temperature using the exponential
             T(x) decay model (sequential propagation, nPro-consistent).

        Stores per-pipe: diameter, velocity, pressure_loss_pa_per_m,
        pressure_loss, heat_loss, delivered_heat, flow_rate, temp_out,
        temp_drop, temp_in, diversity_factor, n_buildings.
        """
        try:
            pl = st.session_state.get("pipe_lengths_m") or {}
            if pl:
                for i, p in enumerate(self.pipes):
                    pid = p.get("id", i)
                    length_m = pl.get(pid)
                    if length_m is not None:
                        p["length_m"] = float(length_m)
        except Exception:
            pass

        try:
            validation_errors = self.validate_system()
            if validation_errors:
                raise ValueError(" | ".join(validation_errors))

            from collections import deque, defaultdict

            def _weak_shortest_path(G, src, dst):
                """BFS on the undirected version of the graph."""
                if src not in G or dst not in G:
                    raise ValueError("Nodes not in graph")
                adj = defaultdict(set)
                for (u, v) in G.edges():
                    adj[u].add(v)
                    adj[v].add(u)
                q = deque([src])
                prev = {src: None}
                while q:
                    u = q.popleft()
                    if u == dst:
                        break
                    for w in adj[u]:
                        if w not in prev:
                            prev[w] = u
                            q.append(w)
                if dst not in prev:
                    raise ValueError("No weak path")
                path = []
                cur = dst
                while cur is not None:
                    path.append(cur)
                    cur = prev[cur]
                path.reverse()
                return path

            def _deg(nid):
                return (self.graph.out_degree(nid) + self.graph.in_degree(nid)) if nid in self.graph else 0

            hub = self.energy_hub
            if hub is None or _deg(hub) == 0:
                eh = st.session_state.get("energy_hubs", {})
                eh_keys_str = {str(k) for k in eh.keys()}
                if eh:
                    for nid, ndata in self.nodes.items():
                        bid = ndata.get("building_id")
                        if (bid in eh or str(bid) in eh_keys_str or ndata.get('type') == 'source') and _deg(nid) > 0:
                            hub = nid
                            break
            if hub is None or _deg(hub) == 0:
                for nid, ndata in self.nodes.items():
                    if ndata.get('type') == 'source' and _deg(nid) > 0:
                        hub = nid
                        break
            if hub is None or _deg(hub) == 0:
                best = max(self.nodes.keys(), key=lambda n: _deg(n), default=None)
                if best and _deg(best) > 0:
                    hub = best

            zero_params = {
                'diameter': 0.0, 'velocity': 0.0,
                'pressure_loss_pa_per_m': 0.0, 'pressure_loss': 0.0,
                'heat_loss': 0.0, 'delivered_heat': 0.0, 'flow_rate': 0.0,
                'temp_out': float(SYSTEM_PARAMS['supply_temp']), 'temp_drop': 0.0,
            }

            if hub is None or _deg(hub) == 0:
                for pipe in self.pipes:
                    pipe['parameters'] = dict(zero_params)
                return

            for pipe in self.pipes:
                pipe['parameters'] = dict(zero_params)

            edge_to_pipe = {}
            for p in self.pipes:
                edge_to_pipe[(p['start_node'], p['end_node'])] = p

            # === 1) Accumulate loads on each pipe ===
            # Residential buildings: space-heating peak (smoothed, peak factor) and
            # number of dwellings are accumulated separately; the DHW peak is added per
            # pipe with its own simultaneity (dhw_peak_kW). Other building types keep the
            # previous smoothed peak (SH + DHW).
            pipe_peak_kW       = defaultdict(float)   # final design load per pipe (kW)
            pipe_sh_peak_kW    = defaultdict(float)   # residential space-heating peak (kW)
            pipe_other_peak_kW = defaultdict(float)   # non-residential peak (kW)
            pipe_dwellings     = defaultdict(int)     # residential dwellings downstream
            pipe_station_kW    = defaultdict(float)   # station power of the building(s) downstream
            pipe_annual_MWh    = defaultdict(float)
            pipe_n_buildings   = defaultdict(int)
            f_safety = float(SYSTEM_PARAMS.get('peak_safety_factor', 1.2))

            # Space-heating peak method (profile from local weather, or fixed factor)
            heat_info = None
            if HEATING_PEAK.get('method') == 'profile':
                loc = self.nodes.get(hub, {}).get('location') if hub else None
                if not loc:
                    loc = st.session_state.get('city_location')
                if loc:
                    heat_info = heating_full_load_hours(float(loc[0]), float(loc[1]),
                                                        float(HEATING_PEAK.get('T_base', 15.0)))
            st.session_state['heating_peak_info'] = (
                {k: v for k, v in heat_info.items() if k != 'profile'} if heat_info else None)
            # (if the weather download fails, the Summary tab reports the factor method)

            def sh_peak_from_annual(sh_MWh_: float, prof_: dict) -> float:
                """Space-heating design peak (kW) of one building from its annual SH demand.
                All buildings share the same normalised degree-hour profile, so the peak of the
                summed profiles equals the sum of the building peaks."""
                if heat_info is not None:
                    return sh_MWh_ * 1000.0 / heat_info['flh'] * float(HEATING_PEAK.get('safety_factor', 1.0))
                return sh_MWh_ * 1000.0 / 8760.0 * float(prof_.get('peak_to_average', 2.5)) * f_safety

            for node_id, ndata in self.nodes.items():
                if ndata.get('type') != 'building':
                    continue

                annual_MWh = float(ndata.get('demand', 0.0) or 0.0)
                b_idx = ndata.get('building_id')
                b_row = None
                if st.session_state.get('b_gdf') is not None and b_idx in st.session_state.b_gdf.index:
                    b_row = st.session_state.b_gdf.loc[b_idx]
                if annual_MWh <= 0 and b_row is not None:
                    annual_MWh = float(
                        (b_row.get('total_demand', 0.0) or 0.0)
                        or (b_row.get('space_heating', 0.0) + b_row.get('domestic_hot_water', 0.0))
                    )
                if annual_MWh <= 0:
                    continue

                b_type = 'Residential'
                if b_row is not None:
                    b_type = b_row.get('building_type', 'Residential') or 'Residential'
                prof = DEMAND_PROFILES.get(b_type, DEMAND_PROFILES['Residential'])

                is_res = (b_type == 'Residential')
                if is_res:
                    sh_MWh = float(b_row.get('space_heating', 0.0) or 0.0) if b_row is not None else 0.0
                    if sh_MWh <= 0:   # no split available -> use profile ratio
                        sh_MWh = annual_MWh * prof['space_heating'] / (prof['space_heating'] + prof['dhw'])
                    sh_peak_kW = sh_peak_from_annual(sh_MWh, prof)
                    fa = float(b_row.get('floor_area', 0.0) or 0.0) if b_row is not None else 0.0
                    n_dw = n_dwellings_of(fa)
                    station_kW = station_power_kW(sh_peak_kW, n_dw)
                else:
                    # non-residential: SH from the same method, DHW as average load
                    _sh = float(b_row.get('space_heating', 0.0) or 0.0) if b_row is not None else 0.0
                    if _sh <= 0:
                        _sh = annual_MWh * prof['space_heating'] / (prof['space_heating'] + prof['dhw'])
                    other_peak_kW = sh_peak_from_annual(_sh, prof) + (annual_MWh - _sh) * 1000.0 / 8760.0

                try:
                    path = _weak_shortest_path(self.graph, hub, node_id)
                except Exception:
                    continue

                for u, v in zip(path[:-1], path[1:]):
                    p = edge_to_pipe.get((u, v)) or edge_to_pipe.get((v, u))
                    if p is None:
                        continue
                    pid = p['id']
                    if is_res:
                        pipe_sh_peak_kW[pid] += sh_peak_kW
                        pipe_dwellings[pid]  += n_dw
                        pipe_station_kW[pid] += station_kW
                    else:
                        pipe_other_peak_kW[pid] += other_peak_kW
                    pipe_annual_MWh[pid]  += annual_MWh
                    pipe_n_buildings[pid] += 1

            # === 1b) Diversity factor (space heating) — Winter et al. (2001) ===
            # DF(n) = A + B / (1 + (n/C)^D)  (Euroheat & Power 09&10/2001, Eq. 6-1, r²=0.95)
            _DF_A = 0.449677646267461
            _DF_B = 0.551234688
            _DF_C = 53.84382392
            _DF_D = 1.762743268

            def winter_diversity_factor(n: int) -> float:
                """Return simultaneity factor for n downstream buildings (Winter et al. 2001)."""
                if n <= 1:
                    return 1.0
                n_clamped = min(n, 200)
                return _DF_A + _DF_B / (1.0 + (n_clamped / _DF_C) ** _DF_D)

            # === 1c) Design load per pipe: SH (with DF) + DHW (own simultaneity) + other types ===
            pipe_diversity_factor = {}
            pipe_dhw_peak_kW = {}
            for pid in set(pipe_sh_peak_kW) | set(pipe_other_peak_kW):
                n = pipe_n_buildings.get(pid, 1)
                df = winter_diversity_factor(n) if APPLY_SH_DIVERSITY else 1.0
                pipe_diversity_factor[pid] = df
                q_dhw = dhw_peak_kW(pipe_dwellings.get(pid, 0))
                pipe_dhw_peak_kW[pid] = q_dhw
                if pipe_n_buildings.get(pid, 0) == 1 and pipe_station_kW.get(pid, 0.0) > 0:
                    # pipe supplies one residential building -> size for its heat transfer station
                    pipe_peak_kW[pid] = pipe_station_kW[pid]
                else:
                    pipe_peak_kW[pid] = (pipe_sh_peak_kW.get(pid, 0.0) + pipe_other_peak_kW.get(pid, 0.0)) * df + q_dhw

            # === 2) Pipe hydraulics + heat losses ===
            rho        = float(SYSTEM_PARAMS['water_density'])
            cp_kJ      = float(SYSTEM_PARAMS['water_specific_heat'])
            dT         = float(SYSTEM_PARAMS['supply_temp'] - SYSTEM_PARAMS['return_temp'])
            T_avg      = 0.5 * (SYSTEM_PARAMS['supply_temp'] + SYSTEM_PARAMS['return_temp'])
            T_soil     = float(SYSTEM_PARAMS.get('ground_temp', 10.0))
            hours      = float(SYSTEM_PARAMS.get('operating_hours', 5000.0))
            nu         = 1.004e-6   # m²/s at ~60 °C
            lambda_ins = float(SYSTEM_PARAMS.get('insulation_lambda', 0.024))
            lambda_soil= float(SYSTEM_PARAMS.get('soil_lambda', 1.2))
            burial     = float(SYSTEM_PARAMS.get('burial_depth', 1.0))
            pair_factor= float(SYSTEM_PARAMS.get('pair_factor', 2.0))
            vmin_      = float(SYSTEM_PARAMS.get('min_velocity', 0.30))
            vmax_      = float(SYSTEM_PARAMS.get('max_velocity', 2.0))
            dns        = sorted(d for d in DN_SIZES if d >= MIN_DN)

            # Build topological order for sequential temperature propagation
            pipe_id_to_pipe = {p['id']: p for p in self.pipes}
            try:
                topo_nodes = list(nx.topological_sort(self.graph))
            except Exception:
                topo_nodes = None

            node_T_out = {hub: float(SYSTEM_PARAMS['supply_temp'])}

            def pipe_topo_key(p):
                sn = p.get('start_node')
                if topo_nodes and sn in topo_nodes:
                    return topo_nodes.index(sn)
                return 9999

            sorted_pipes = sorted(self.pipes, key=pipe_topo_key)

            for pipe in sorted_pipes:
                pid          = pipe['id']
                peak_kW      = float(pipe_peak_kW.get(pid, 0.0))
                delivered_MWh= float(pipe_annual_MWh.get(pid, 0.0))
                length_m     = float(pipe.get('length_m', 0.0))
                material     = pipe.get('material', SYSTEM_PARAMS['default_material'])
                pipe_type    = pipe.get('type', 'main')
                start_node   = pipe.get('start_node')
                T_in_seq     = node_T_out.get(start_node, float(SYSTEM_PARAMS['supply_temp']))

                if peak_kW <= 0.0 or length_m <= 0.0:
                    pipe['parameters']['delivered_heat'] = delivered_MWh
                    continue

                # Thermal load → mass flow → volume flow
                m_dot = peak_kW / (cp_kJ * dT)
                q_m3s = m_dot / rho

                # nPro-style DN selection: smallest DN in recommended dp/m band
                roughness = PIPE_MATERIALS.get(material, PIPE_MATERIALS['Steel'])['roughness']
                DP_MIN, DP_MAX = 30.0, 150.0
                best = None

                for dn in dns:
                    d_m_try, v_try, Re_try, f_try, dp_m_try = pipe_hydraulics(dn, q_m3s, nu, rho, roughness)
                    if v_try > vmax_:
                        continue
                    if DP_MIN <= dp_m_try <= DP_MAX:
                        best = (dn, d_m_try, v_try, Re_try, f_try, dp_m_try)
                        break

                if best is None:
                    for dn in dns:
                        d_m_try, v_try, Re_try, f_try, dp_m_try = pipe_hydraulics(dn, q_m3s, nu, rho, roughness)
                        if v_try <= vmax_:
                            best = (dn, d_m_try, v_try, Re_try, f_try, dp_m_try)
                            break

                if best is None:
                    dn = dns[-1]
                    best = (dn,) + pipe_hydraulics(dn, q_m3s, nu, rho, roughness)

                dn_selected, d_m, v, Re, f, dp_per_m = best
                dp_total = dp_per_m * length_m

                annual_heat_loss_MWh, T_out, dT_seg = pipe_heat_loss_and_temp(
                    T_avg, T_soil, length_m, hours,
                    d_m, material, rho, q_m3s,
                    lambda_ins, lambda_soil, burial, pair_factor, pipe_type,
                    cp_kJ, T_in_override=T_in_seq, dn=dn_selected
                )

                end_node = pipe.get('end_node')
                if end_node is not None:
                    node_T_out[end_node] = T_out

                pipe['parameters'] = {
                    'diameter':               float(dn_selected),
                    'velocity':               float(v),
                    'pressure_loss_pa_per_m': float(dp_per_m),
                    'pressure_loss':          float(dp_total),
                    'heat_loss':              float(annual_heat_loss_MWh),
                    'delivered_heat':         float(delivered_MWh),
                    'flow_rate':              float(q_m3s * 86400.0),
                    'temp_out':               float(T_out),
                    'temp_drop':              float(dT_seg),
                    'temp_in':                float(T_in_seq),
                    'diversity_factor':       float(pipe_diversity_factor.get(pid, 1.0)),
                    'n_buildings':            int(pipe_n_buildings.get(pid, 0)),
                    'n_dwellings':            int(pipe_dwellings.get(pid, 0)),
                    'design_load_kW':         float(peak_kW),
                    'dhw_peak_kW':            float(pipe_dhw_peak_kW.get(pid, 0.0)),
                }

            # Auto-prune dead pipes and orphan junctions
            removed = self.prune_zero_flow_pipes(flow_eps_m3day=0.01, heat_eps_MWh=1e-3)
            if removed:
                self.prune_isolated_junctions()

        except Exception as e:
            st.error(f"Flow calculation error: {str(e)}")
            raise

    def diagnose_connectivity(self) -> list:
        """Return building_ids that have no weak path from the hub.

        Returns:
            List of building_id values for disconnected buildings.
        """
        def _deg(nid):
            return (self.graph.out_degree(nid) + self.graph.in_degree(nid)) if nid in self.graph else 0

        hub = self.energy_hub
        if hub is None or _deg(hub) == 0:
            eh = st.session_state.get("energy_hubs", {})
            if eh:
                for nid, ndata in self.nodes.items():
                    if ndata.get("building_id") in eh and _deg(nid) > 0:
                        hub = nid
                        break
        if hub is None or _deg(hub) == 0:
            return [ndata.get('building_id') for nid, ndata in self.nodes.items()
                    if ndata.get('type') == 'building']

        from collections import defaultdict, deque
        adj = defaultdict(set)
        for (u, v) in self.graph.edges():
            adj[u].add(v)
            adj[v].add(u)

        def has_path(s, t):
            if s not in adj or t not in adj:
                return False
            seen = {s}
            dq = deque([s])
            while dq:
                u = dq.popleft()
                if u == t:
                    return True
                for w in adj[u]:
                    if w not in seen:
                        seen.add(w)
                        dq.append(w)
            return False

        disconnected = []
        for nid, ndata in self.nodes.items():
            if ndata.get('type') == 'building':
                if not has_path(hub, nid):
                    disconnected.append(ndata.get('building_id'))
        return disconnected

    def get_pipe_by_id(self, pipe_id: int):
        """Return the pipe dict with the given ID, or None if not found.

        Args:
            pipe_id: Integer pipe ID.

        Returns:
            Pipe dict or None.
        """
        for pipe in self.pipes:
            if pipe['id'] == pipe_id:
                return pipe
        return None

    def auto_connect_buildings(self, buildings_gdf: gpd.GeoDataFrame,
                                max_distance_m: float = None) -> int:
        """Auto-connect unconnected buildings to the nearest main pipe.

        For each candidate building (positive demand, not yet connected):
          1. Finds the nearest main pipe in projected coordinates.
          2. Inserts a junction node at the connection point (splits the pipe).
          3. Adds a 'connection' type pipe from junction to building node.

        Args:
            buildings_gdf:  GeoDataFrame of buildings.
            max_distance_m: Optional maximum lateral distance in metres;
                            buildings farther than this are skipped.

        Returns:
            Number of buildings successfully connected.
        """
        from shapely.geometry import shape, Point, LineString, mapping

        if 'connected_buildings' not in st.session_state:
            st.session_state.connected_buildings = {}

        if not self.pipes:
            raise ValueError("No pipes in the network to connect to")

        def crs_is_degrees(gdf):
            try:
                ai = getattr(gdf.crs, "axis_info", None)
                return bool(ai and any(getattr(a, "unit_name", "").lower() in ("degree", "degrees") for a in ai))
            except Exception:
                return False

        is_deg = crs_is_degrees(buildings_gdf)
        if is_deg:
            st.warning("Your layer appears to be in degrees (lat/lon). Distances in meters will be approximate.")

        tol = 1.0 if not is_deg else (1.0 / 111_320.0)

        candidates = []
        for idx, row in buildings_gdf.iterrows():
            if idx in st.session_state.connected_buildings:
                continue
            demand = float(row.get('total_demand', 0.0) or 0.0)
            if demand <= 0:
                continue
            geom = row.geometry
            pt = geom if isinstance(geom, Point) else geom.centroid
            candidates.append({'id': idx, 'point': pt, 'geom': geom, 'demand': demand})

        if not candidates:
            st.info("No unconnected buildings with positive demand found.")
            return 0

        main_pipes = [p for p in self.pipes if p.get('type') == 'main']
        if not main_pipes:
            raise ValueError("No main pipes to connect to. Please draw a main pipe first.")

        made = 0

        for b in candidates:
            main_pipes = [p for p in self.pipes if p.get('type') == 'main']
            if not main_pipes:
                break
            try:
                junc = None
                junc_xy = None
                nearest_pipe, connection_point, min_dist = None, None, float('inf')
                proj_epsg = None
                try:
                    if st.session_state.get("metric_crs") is not None:
                        proj_epsg = int(st.session_state.metric_crs.to_epsg() or 3857)
                except Exception:
                    proj_epsg = None

                for p in main_pipes:
                    try:
                        line = shape(p['geometry'])
                        pt = b['geom']  # building outline (as in nPro); a Point if no polygon
                        if proj_epsg is not None:
                            gline = gpd.GeoSeries([line], crs="EPSG:4326").to_crs(epsg=proj_epsg).iloc[0]
                            gpt = gpd.GeoSeries([pt], crs="EPSG:4326").to_crs(epsg=proj_epsg).iloc[0]
                            d = float(gline.distance(gpt))
                            near_proj = nearest_points(gline, gpt)[0]
                            near_geo = gpd.GeoSeries([near_proj], crs=f"EPSG:{proj_epsg}").to_crs(epsg=4326).iloc[0]
                            candidate_conn = near_geo
                        else:
                            d = float(pt.distance(line))
                            candidate_conn = nearest_points(line, pt)[0]
                        if d < min_dist:
                            min_dist = d
                            nearest_pipe = p
                            connection_point = candidate_conn
                    except Exception:
                        continue

                if nearest_pipe is None:
                    continue

                if max_distance_m is not None:
                    if proj_epsg is not None:
                        min_dist_m = float(min_dist)
                    else:
                        meters_per_deg = 111_320.0 * math.cos(math.radians(b['point'].y))
                        min_dist_m = float(min_dist) * meters_per_deg
                    if min_dist_m > max_distance_m:
                        continue

                orig_line = shape(nearest_pipe['geometry'])
                coords    = list(orig_line.coords)
                start_pt  = Point(coords[0])
                end_pt    = Point(coords[-1])

                if proj_epsg is not None:
                    s_proj    = gpd.GeoSeries([start_pt],        crs="EPSG:4326").to_crs(epsg=proj_epsg).iloc[0]
                    e_proj    = gpd.GeoSeries([end_pt],          crs="EPSG:4326").to_crs(epsg=proj_epsg).iloc[0]
                    conn_proj = gpd.GeoSeries([connection_point], crs="EPSG:4326").to_crs(epsg=proj_epsg).iloc[0]
                    d_to_start = float(conn_proj.distance(s_proj))
                    d_to_end   = float(conn_proj.distance(e_proj))
                else:
                    d_to_start = float(connection_point.distance(start_pt))
                    d_to_end   = float(connection_point.distance(end_pt))

                eps = 0.5 if proj_epsg is not None else 0.000005

                if d_to_start <= eps:
                    reuse_node = nearest_pipe['start_node']
                    reuse_xy   = (start_pt.x, start_pt.y)
                    connection_point = start_pt
                    junc   = reuse_node
                    junc_xy = reuse_xy
                elif d_to_end <= eps:
                    reuse_node = nearest_pipe['end_node']
                    reuse_xy   = (end_pt.x, end_pt.y)
                    connection_point = end_pt
                    junc   = reuse_node
                    junc_xy = reuse_xy
                else:
                    cum = [0.0]
                    for i in range(len(coords) - 1):
                        seg = LineString([coords[i], coords[i + 1]])
                        cum.append(cum[-1] + seg.length)
                    total_len = cum[-1] if cum else 0.0

                    if total_len == 0:
                        if d_to_start <= d_to_end:
                            reuse_node = nearest_pipe['start_node']
                            reuse_xy   = (coords[0][0], coords[0][1])
                            connection_point = start_pt
                            junc = reuse_node; junc_xy = reuse_xy
                        else:
                            reuse_node = nearest_pipe['end_node']
                            reuse_xy   = (coords[-1][0], coords[-1][1])
                            connection_point = end_pt
                            junc = reuse_node; junc_xy = reuse_xy
                    else:
                        s_param = orig_line.project(connection_point)
                        insert_idx = None
                        for i in range(len(cum) - 1):
                            if s_param >= cum[i] - 1e-9 and s_param <= cum[i + 1] + 1e-9:
                                insert_idx = i
                                break

                        if insert_idx is None:
                            if d_to_start <= d_to_end:
                                reuse_node = nearest_pipe['start_node']
                                reuse_xy   = (coords[0][0], coords[0][1])
                                connection_point = start_pt
                            else:
                                reuse_node = nearest_pipe['end_node']
                                reuse_xy   = (coords[-1][0], coords[-1][1])
                                connection_point = end_pt
                            junc    = reuse_node
                            junc_xy = reuse_xy
                        else:
                            cx, cy = (connection_point.x, connection_point.y)
                            existing_coords = [(round(x, 12), round(y, 12)) for (x, y) in coords]
                            cand = (round(cx, 12), round(cy, 12))

                            if cand in existing_coords:
                                pos = existing_coords.index(cand)
                                if pos == 0:
                                    reuse_node = nearest_pipe['start_node']
                                    reuse_xy   = (coords[0][0], coords[0][1])
                                    connection_point = Point(coords[0])
                                elif pos == len(coords) - 1:
                                    reuse_node = nearest_pipe['end_node']
                                    reuse_xy   = (coords[-1][0], coords[-1][1])
                                    connection_point = Point(coords[-1])
                                else:
                                    junc_xy_ = coords[pos]
                                    seg1 = LineString(coords[:pos + 1])
                                    seg2 = LineString(coords[pos:])
                                    # Safety: degenerate segment → snap to nearest endpoint
                                    if seg1.length < 1e-9 or seg2.length < 1e-9:
                                        if d_to_start <= d_to_end:
                                            junc    = nearest_pipe['start_node']
                                            junc_xy = (coords[0][0], coords[0][1])
                                        else:
                                            junc    = nearest_pipe['end_node']
                                            junc_xy = (coords[-1][0], coords[-1][1])
                                        reuse_node       = junc
                                        connection_point = Point(junc_xy)
                                    else:
                                        junc = self.add_node('junction', (junc_xy_[1], junc_xy_[0]))
                                        junc_xy = (junc_xy_[0], junc_xy_[1])
                                        self.add_pipe_segment(nearest_pipe['start_node'], junc,
                                                              mapping(seg1), nearest_pipe.get('material'), 'main')
                                        self.add_pipe_segment(junc, nearest_pipe['end_node'],
                                                              mapping(seg2), nearest_pipe.get('material'), 'main')
                                        if self.graph.has_edge(nearest_pipe['start_node'], nearest_pipe['end_node']):
                                            self.graph.remove_edge(nearest_pipe['start_node'], nearest_pipe['end_node'])
                                        try:
                                            self.pipes.remove(nearest_pipe)
                                        except ValueError:
                                            if 'id' in nearest_pipe:
                                                self.pipes[:] = [pp for pp in self.pipes if pp.get('id') != nearest_pipe['id']]
                                        reuse_node       = junc
                                        connection_point = Point(junc_xy)
                            else:
                                new_coords = coords[:insert_idx + 1] + [(cx, cy)] + coords[insert_idx + 1:]
                                seg1_coords = new_coords[:insert_idx + 2]
                                seg2_coords = new_coords[insert_idx + 1:]
                                seg1 = LineString(seg1_coords)
                                seg2 = LineString(seg2_coords)
                                # Safety: if either segment is degenerate, snap to nearest endpoint
                                # instead of splitting — this prevents the original pipe being deleted
                                # with nothing to replace it.
                                if seg1.length < 1e-9 or seg2.length < 1e-9:
                                    if d_to_start <= d_to_end:
                                        junc = nearest_pipe['start_node']
                                        junc_xy = (coords[0][0], coords[0][1])
                                    else:
                                        junc = nearest_pipe['end_node']
                                        junc_xy = (coords[-1][0], coords[-1][1])
                                    reuse_node = junc
                                    reuse_xy   = junc_xy
                                    connection_point = Point(junc_xy)
                                else:
                                    junc = self.add_node('junction', (cy, cx))
                                    junc_xy = (cx, cy)
                                    self.add_pipe_segment(nearest_pipe['start_node'], junc,
                                                          mapping(seg1), nearest_pipe.get('material'), 'main')
                                    self.add_pipe_segment(junc, nearest_pipe['end_node'],
                                                          mapping(seg2), nearest_pipe.get('material'), 'main')
                                    if self.graph.has_edge(nearest_pipe['start_node'], nearest_pipe['end_node']):
                                        self.graph.remove_edge(nearest_pipe['start_node'], nearest_pipe['end_node'])
                                    try:
                                        self.pipes.remove(nearest_pipe)
                                    except ValueError:
                                        if 'id' in nearest_pipe:
                                            self.pipes[:] = [pp for pp in self.pipes if pp.get('id') != nearest_pipe['id']]
                                    if 'id' in nearest_pipe:
                                        main_pipes = [pp for pp in main_pipes if pp.get('id') != nearest_pipe['id']]
                                    reuse_node = junc
                                    reuse_xy   = (cx, cy)
                                    connection_point = Point(cx, cy)

                b_node = self.add_node('building', (b['point'].y, b['point'].x), b['demand'], b['id'])
                # Connection ends at the building outline (point nearest to the junction), as in nPro.
                # Falls back to the centroid if the junction lies inside the building.
                end_pt = b['point']
                try:
                    g = b['geom']
                    if not isinstance(g, Point):
                        jpt = Point(junc_xy)
                        if proj_epsg is not None:
                            gg = gpd.GeoSeries([g], crs="EPSG:4326").to_crs(epsg=proj_epsg).iloc[0]
                            gj = gpd.GeoSeries([jpt], crs="EPSG:4326").to_crs(epsg=proj_epsg).iloc[0]
                            if not gg.contains(gj):
                                on_wall = nearest_points(gg, gj)[0]
                                if on_wall.distance(gj) > 0.5:
                                    end_pt = gpd.GeoSeries([on_wall], crs=f"EPSG:{proj_epsg}").to_crs(epsg=4326).iloc[0]
                        elif not g.contains(jpt):
                            on_wall = nearest_points(g, jpt)[0]
                            if on_wall.distance(jpt) > 1e-6:
                                end_pt = on_wall
                except Exception:
                    end_pt = b['point']

                b_node = self.add_node('building', (end_pt.y, end_pt.x), b['demand'], b['id'])

                start_xy = junc_xy
                connect_line = LineString([start_xy, (end_pt.x, end_pt.y)])
                if connect_line.length == 0:
                    st.warning(f"Skipped building {b['id']} due to zero-length connection.")
                    continue

                self.add_pipe_segment(
                    start_node=junc,
                    end_node=b_node,
                    geometry=mapping(connect_line),
                    material=SYSTEM_PARAMS['default_material'],
                    pipe_type='connection'
                )

                st.session_state.connected_buildings[b['id']] = b_node
                made += 1

            except Exception as e:
                st.warning(f"Could not connect building {b['id']}: {str(e)}")
                continue

        return made



# ============================================================
# SECTION: Project Save / Load
# ============================================================

def save_project() -> str:
    """Serialize the current session state to a JSON string for download.

    Returns:
        JSON string containing buildings GDF, selected areas, energy hubs,
        city location, draw mode, pipe network nodes/pipes, and connected buildings.
    """
    data = {}

    if st.session_state.b_gdf is not None:
        data['b_gdf'] = json.loads(st.session_state.b_gdf.to_json())
    else:
        data['b_gdf'] = None

    data['selected_areas']       = st.session_state.get('selected_areas', [])
    data['energy_hubs']          = st.session_state.get('energy_hubs', {})
    data['city_location']        = st.session_state.get('city_location', None)
    data['draw_mode']            = st.session_state.get('draw_mode', 'Select Area')

    pn = st.session_state.pipe_network
    data['pipe_network'] = {
        'nodes': pn.nodes,
        'pipes': pn.pipes,
    }

    data['connected_buildings'] = st.session_state.get('connected_buildings', {})

    return json.dumps(data)


def load_project(json_str: str):
    """Restore session state from a JSON string produced by save_project().

    Args:
        json_str: JSON string as returned by save_project().
    """
    data = json.loads(json_str)

    if data.get('b_gdf'):
        gdf = gpd.GeoDataFrame.from_features(data['b_gdf']['features'])
        gdf.crs = "EPSG:4326"
        st.session_state.b_gdf = gdf
    else:
        st.session_state.b_gdf = None

    st.session_state.selected_areas     = data.get('selected_areas', [])
    st.session_state.energy_hubs        = data.get('energy_hubs', {})
    st.session_state.city_location      = data.get('city_location', None)
    # Restore map center to the project's location so it opens in the right place
    if st.session_state.city_location:
        st.session_state.map_center = list(st.session_state.city_location)
        st.session_state.map_zoom   = 14
    st.session_state.draw_mode          = data.get('draw_mode', 'Select Area')
    st.session_state.connected_buildings= data.get('connected_buildings', {})

    pn_data = data.get('pipe_network', {})
    pn = PipeNetwork()
    pn.nodes = pn_data.get('nodes', {})
    pn.pipes = pn_data.get('pipes', [])
    st.session_state.pipe_network = pn

    st.session_state.map_rev += 1


def show_npro_network_creator():
    """Render the nPro-style star network creation UI in the sidebar."""
    if st.session_state.b_gdf is not None:
        st.subheader("nPro-style Network Creation")

        if 'is_energy_hub' not in st.session_state.b_gdf.columns:
            st.session_state.b_gdf['is_energy_hub'] = False

        hub_index = st.selectbox(
            "Select Energy Hub Building",
            st.session_state.b_gdf.index,
            format_func=lambda x: (
                f"{st.session_state.b_gdf.loc[x, 'address']} "
                f"(Area: {st.session_state.b_gdf.loc[x, 'floor_area']:.0f}m²)"
            )
        )
        st.session_state.b_gdf['is_energy_hub'] = False
        st.session_state.b_gdf.loc[hub_index, 'is_energy_hub'] = True

        if st.button("Create nPro Star Network"):
            try:
                st.session_state.pipe_network.auto_connect_star_network(st.session_state.b_gdf)
                st.success("Created nPro-style star network!")
                st.rerun()
            except Exception as e:
                st.error(f"Error creating network: {str(e)}")


# ============================================================
# SECTION: UI / Sidebar — Session State Init & App Config
# ============================================================

# Streamlit page configuration
st.set_page_config(
    page_title="DHN Planner",
    page_icon="🔥",
    layout="wide",
)

# ── Global dark-sidebar + modern card CSS ────────────────────────────────────
st.markdown("""
<style>
/* ── Sidebar dark theme ── */
[data-testid="stSidebar"] {
    background: #1e2130 !important;
}
[data-testid="stSidebar"] * {
    color: #e8eaf0 !important;
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stRadio label,
[data-testid="stSidebar"] .stCheckbox label,
[data-testid="stSidebar"] .stNumberInput label {
    color: #b0b8d0 !important;
    font-size: 12px !important;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #ffffff !important;
    border-left: 3px solid #27ae60;
    padding-left: 8px;
}
[data-testid="stSidebar"] .stSelectbox > div > div {
    background: #2d3348 !important;
    border-color: #3d4460 !important;
    color: #e8eaf0 !important;
}
[data-testid="stSidebar"] hr {
    border-color: #3d4460 !important;
}
/* ── App header ── */
.dhn-header {
    background: linear-gradient(135deg, #1a2540 0%, #1e3a5f 60%, #0d5c3a 100%);
    border-radius: 12px;
    padding: 18px 28px;
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 16px;
}
.dhn-header h1 {
    color: #ffffff;
    margin: 0;
    font-size: 26px;
    font-weight: 700;
    letter-spacing: 0.02em;
}
.dhn-header p {
    color: #94b8e0;
    margin: 4px 0 0 0;
    font-size: 13px;
}
/* ── Step progress bar ── */
.step-bar {
    display: flex;
    align-items: center;
    background: #f7f9fc;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 12px 20px;
    margin-bottom: 14px;
    gap: 0;
}
.step-item {
    display: flex;
    align-items: center;
    flex: 1;
    min-width: 0;
}
.step-circle {
    width: 28px;
    height: 28px;
    border-radius: 50%;
    background: #27ae60;
    color: #fff;
    font-weight: 700;
    font-size: 13px;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    box-shadow: 0 2px 6px rgba(39,174,96,0.4);
}
.step-text {
    font-size: 12px;
    color: #445;
    margin-left: 8px;
    line-height: 1.3;
}
.step-text b { color: #1a2540; }
.step-connector {
    width: 24px;
    height: 2px;
    background: #cbd5e0;
    flex-shrink: 0;
    margin: 0 2px;
}
/* ── KPI / results card ── */
.kpi-card {
    background: #fff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 14px 18px;
    box-shadow: 0 1px 6px rgba(0,0,0,0.06);
    margin-bottom: 10px;
}
.kpi-card h4 {
    color: #1a2540;
    margin: 0 0 10px 0;
    font-size: 14px;
    font-weight: 700;
    border-bottom: 2px solid #27ae60;
    padding-bottom: 6px;
}
/* ── Results table nPro-style ── */
.res-table { width: 100%; border-collapse: collapse; font-size: 13px; }
.res-table th {
    background: #1e2130;
    color: #fff;
    padding: 6px 10px;
    text-align: right;
    font-weight: 600;
}
.res-table th:first-child { text-align: left; }
.res-table td {
    padding: 5px 10px;
    border-bottom: 1px solid #f0f0f0;
    color: #334;
    text-align: right;
}
.res-table td:first-child { text-align: left; color: #556; }
.res-table tr:hover td { background: #f7faff; }
.res-table tr.total-row td { font-weight: 700; background: #f0f7ff; color: #1a2540; }
.res-table tr.section-row td { font-weight: 700; color: #27ae60; background: #f5fff8; }
/* ── Building info card ── */
.bld-card {
    background: #fff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 14px 16px;
    margin-bottom: 8px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
}
.bld-label { font-size: 11px; color: #8899aa; text-transform: uppercase; letter-spacing: 0.05em; }
.bld-value { font-size: 20px; font-weight: 700; color: #1a2540; }
.bld-unit  { font-size: 12px; color: #8899aa; margin-left: 3px; }
/* ── Demand bar ── */
.demand-bar-wrap {
    background: #f0f0f0;
    border-radius: 4px;
    height: 8px;
    width: 100%;
    margin: 4px 0 10px 0;
    overflow: hidden;
}
.demand-bar-sh  { height: 100%; background: #e67e22; float: left; }
.demand-bar-dhw { height: 100%; background: #3498db; float: left; }
/* ── Section divider with label ── */
.section-label {
    font-size: 11px;
    font-weight: 700;
    color: #27ae60;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin: 14px 0 6px 0;
    padding-left: 2px;
}
</style>

<div class="dhn-header">
  <div style="font-size:36px;line-height:1">🔥</div>
  <div>
    <h1>District Heating Network Planner</h1>
    <p>Open-source DHN dimensioning tool &nbsp;·&nbsp; Hydraulic &amp; thermal simulation &nbsp;·&nbsp; OSM-based building extraction</p>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Step progress bar ────────────────────────────────────────────────────────
st.markdown("""
<div class="step-bar">
  <div class="step-item">
    <div class="step-circle">1</div>
    <div class="step-text"><b>Select Area</b><br>extract buildings</div>
  </div>
  <div class="step-connector"></div>
  <div class="step-item">
    <div class="step-circle">2</div>
    <div class="step-text"><b>Draw Building</b><br>add manually</div>
  </div>
  <div class="step-connector"></div>
  <div class="step-item">
    <div class="step-circle">3</div>
    <div class="step-text"><b>Draw Main Pipe</b><br>route backbone</div>
  </div>
  <div class="step-connector"></div>
  <div class="step-item">
    <div class="step-circle">4</div>
    <div class="step-text"><b>Auto-Connect</b><br>then set hub</div>
  </div>
  <div class="step-connector"></div>
  <div class="step-item">
    <div class="step-circle">5</div>
    <div class="step-text"><b>Recalculate</b><br>run simulation</div>
  </div>
</div>
""", unsafe_allow_html=True)

# Initialize session state variables that persist across user interactions
if 'b_gdf' not in st.session_state:
    st.session_state.b_gdf = None
if 'selected_building' not in st.session_state:
    st.session_state.selected_building = None
if 'connected_buildings' not in st.session_state:
    st.session_state.connected_buildings = {}
if 'energy_hubs' not in st.session_state:
    st.session_state.energy_hubs = {}
if 'pipe_network' not in st.session_state:
    st.session_state.pipe_network = PipeNetwork()
if 'building_clusters' not in st.session_state:
    st.session_state.building_clusters = {}
if 'city_location' not in st.session_state:
    st.session_state.city_location = None
if 'draw_mode' not in st.session_state:
    st.session_state.draw_mode = "Select Area"
if 'selected_pipe_idx' not in st.session_state:
    st.session_state.selected_pipe_idx = None
if 'manual_building_counter' not in st.session_state:
    st.session_state.manual_building_counter = 1
if 'selected_areas' not in st.session_state:
    st.session_state.selected_areas = []
if 'selected_pipe' not in st.session_state:
    st.session_state.selected_pipe = None
if 'map_rev' not in st.session_state:
    st.session_state.map_rev = 0


# ============================================================
# SECTION: Main App Logic
# ============================================================

city = st.text_input("Enter City (e.g., 'Paris, France', 'New York, USA')", "Ingolstadt, Germany")

if city and city != st.session_state.get("_last_geocoded_city"):
    try:
        with st.spinner(f"Locating {city}..."):
            loc = ox.geocode_to_gdf(city)
            centroid = loc.to_crs(epsg=4326).geometry.centroid.iloc[0]
            st.session_state.city_location = [centroid.y, centroid.x]
            st.session_state.map_center = [centroid.y, centroid.x]
            st.session_state.map_zoom = 14
            st.session_state._last_geocoded_city = city
            st.session_state._map_locked_to_project = False  # allow re-center when buildings load

            if len(st.session_state.pipe_network.nodes) == 0:
                st.session_state.pipe_network.add_node('source', st.session_state.city_location, 0)
    except Exception as e:
        st.error(f"Could not locate city: {e}")
        st.session_state.city_location = None

if st.session_state.city_location:
    city_lat, city_lon = st.session_state.city_location

    # ── Project Save / Load ──────────────────────────────────────────────
    with st.expander("💾 Save / Load Project", expanded=False):
        col_s, col_l = st.columns(2)
        with col_s:
            project_json = save_project()
            st.download_button(
                label="💾 Save Project",
                data=project_json,
                file_name="dhn_project.json",
                mime="application/json",
            )
        with col_l:
            uploaded = st.file_uploader("Load Project (.json)", type="json", key="project_upload")
            if uploaded is not None:
                # Use file name + size as a signature to avoid re-loading on every rerun
                _upload_sig = f"{uploaded.name}_{uploaded.size}"
                if st.session_state.get("_last_loaded_project") != _upload_sig:
                    st.session_state["_last_loaded_project"] = _upload_sig
                    load_project(uploaded.read().decode("utf-8"))
                    st.session_state._last_geocoded_city = city  # prevent geocoder overwriting loaded project
                    # Center map on the project's city location
                    if st.session_state.city_location:
                        st.session_state.map_center = list(st.session_state.city_location)
                        st.session_state.map_zoom = 14
                    st.success("Project loaded!")
                    st.rerun()

    # ── Primary action buttons — highlighted so users follow the correct sequence ──
    # These two buttons are the core workflow: connect buildings → run simulation.
    # They use custom CSS to stand out from all other controls.
    st.markdown("""
    <style>
    div[data-testid="stButton"] > button[kind="primary"] {
        background: linear-gradient(135deg, #1a6b3c 0%, #27ae60 100%);
        color: #fff;
        border: none;
        border-radius: 8px;
        font-size: 15px;
        font-weight: 700;
        padding: 0.55rem 1.2rem;
        letter-spacing: 0.03em;
        box-shadow: 0 2px 8px rgba(39,174,96,0.35);
        width: 100%;
    }
    div[data-testid="stButton"] > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #145a32 0%, #1e8449 100%);
        box-shadow: 0 4px 14px rgba(39,174,96,0.50);
    }
    </style>
    """, unsafe_allow_html=True)

    _action_c1, _action_c2 = st.columns(2)
    with _action_c1:
        st.markdown("**Step ①** — Connect buildings to main pipe(s)")
        st.caption("Safe to re-run after drawing more pipes or selecting more areas.")
        if st.button("🔗 Connect New Buildings", type="primary", key="btn_autoconnect_main"):
            if st.session_state.get('b_gdf') is not None:
                try:
                    n = st.session_state.pipe_network.auto_connect_buildings(
                        st.session_state.b_gdf, max_distance_m=1000
                    )
                    if n > 0:
                        st.success(f"Connected {n} new building(s) to the network!")
                    else:
                        st.info("All buildings are already connected.")
                    st.session_state.map_rev += 1
                    st.rerun()
                except Exception as e:
                    st.error(str(e))
            else:
                st.error("No buildings available to connect.")

    with _action_c2:
        st.markdown("**Step ②** — Run hydraulic & thermal simulation")
        if st.button("▶ Recalculate Flows", type="primary", key="btn_recalc_main"):
            has_hub = any(nd.get('type') == 'source'
                          for nd in st.session_state.pipe_network.nodes.values())
            if not has_hub:
                st.error("Please set an Energy Hub first (click a building → 'Set as Energy Hub').")
            else:
                errs = st.session_state.pipe_network.validate_system()
                if errs:
                    for err in errs:
                        st.error(f"• {err}")
                else:
                    try:
                        with st.spinner("Calculating flows…"):
                            st.session_state.pipe_network.calculate_network_flows()
                        st.session_state.flow_running = False
                        st.session_state.flow_done = True
                        st.session_state.flow_error = None
                        st.success("Flows updated.")
                        st.session_state._needs_redraw = True
                        st.session_state.map_rev += 1
                        st.rerun()
                    except Exception as e:
                        import traceback
                        st.error(f"Calculation error: {str(e)}")
                        st.code(traceback.format_exc())

    # Step ③ — remove main pipes that carry zero flow after calculation
    with st.expander("🧹 Cleanup — Remove zero-flow pipes", expanded=False):
        st.caption("After recalculating flows, remove main pipe segments that carry no heat (no buildings connected). Connection pipes are never removed here.")
        if st.button("🗑 Remove Unused Main Pipes", key="btn_remove_zero_pipes"):
            pn = st.session_state.pipe_network
            before = len(pn.pipes)
            # Only remove main pipes whose flow_rate == 0 after a completed calculation
            if not st.session_state.get('flow_done'):
                st.warning("Please recalculate flows first so zero-flow pipes can be identified.")
            else:
                zero_main = [
                    p for p in pn.pipes
                    if p.get('type') == 'main'
                    and float((p.get('parameters') or {}).get('flow_rate', 0) or 0) == 0.0
                ]
                if not zero_main:
                    st.info("No zero-flow main pipes found.")
                else:
                    for p in zero_main:
                        # Remove from graph
                        sn, en = p.get('start_node'), p.get('end_node')
                        if sn and en and pn.graph.has_edge(sn, en):
                            pn.graph.remove_edge(sn, en)
                        try:
                            pn.pipes.remove(p)
                        except ValueError:
                            if 'id' in p:
                                pn.pipes[:] = [pp for pp in pn.pipes if pp.get('id') != p['id']]
                    after = len(pn.pipes)
                    removed = before - after
                    st.success(f"Removed {removed} zero-flow main pipe(s).")
                    st.session_state.map_rev += 1
                    st.rerun()

    # Network stats and controls are in the right sidebar — no duplicate expander here

    # ── Right sidebar (Streamlit native sidebar) ─────────────────────────
    with st.sidebar:
        st.markdown("""
<style>
section[data-testid="stSidebar"] {
    background: #1e2130 !important;
    min-width: 280px !important;
}
section[data-testid="stSidebar"] > div { background: #1e2130 !important; }
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] div { color: #c8d0e0 !important; }
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #ffffff !important;
    border-left: 3px solid #27ae60;
    padding-left: 8px;
    margin-top: 16px !important;
}
section[data-testid="stSidebar"] .stSelectbox > div > div,
section[data-testid="stSidebar"] .stTextInput > div > div input {
    background: #2d3348 !important;
    border-color: #3d4460 !important;
    color: #e8eaf0 !important;
}
section[data-testid="stSidebar"] .stRadio > div { gap: 4px; }
section[data-testid="stSidebar"] hr { border-color: #3d4460 !important; }
section[data-testid="stSidebar"] .stCheckbox label { font-size: 13px !important; }
/* Danger button red */
section[data-testid="stSidebar"] button[kind="secondary"] {
    background: #2d3348 !important;
    border-color: #3d4460 !important;
    color: #c8d0e0 !important;
    border-radius: 6px !important;
}
</style>
""", unsafe_allow_html=True)

        # ── Tool header ──
        st.markdown('<div style="text-align:center;padding:10px 0 4px"><span style="font-size:28px">🔥</span><br><span style="color:#fff;font-weight:700;font-size:15px">DHN Planner</span></div>', unsafe_allow_html=True)
        st.markdown("---")

        # ── Drawing Mode ──
        st.subheader("Drawing Mode")
        _draw_opts = ["Select Area", "Add Building", "Add Main Pipe"]
        _draw_idx = _draw_opts.index(st.session_state.draw_mode) if st.session_state.draw_mode in _draw_opts else 0
        _chosen_mode = st.radio(
            "Select drawing purpose:",
            _draw_opts,
            index=_draw_idx,
            key="draw_mode_radio",
            help=(
                "Select Area: Draw to extract OSM buildings. "
                "Add Building: Draw to create fictional buildings. "
                "Add Main Pipe: Draw the main distribution pipe."
            ),
        )
        if _chosen_mode != st.session_state.draw_mode:
            st.session_state.draw_mode = _chosen_mode
            st.rerun()
        st.markdown("---")

        # ── Map styling ──
        # NOTE: use key= only — never also assign to st.session_state directly,
        # as that causes a rerun feedback loop on every render.
        st.subheader("Map Styling")
        _pm_opts = list(PIPE_STYLE_CHOICES.keys())
        st.selectbox("Pipe color / label", _pm_opts, index=0, key="pipe_metric")
        _bl_opts = [
            "Total heat demand (MWh/a)",
            "Space heating (MWh/a)",
            "Hot water demand (MWh/a)",
            "Floor area (m²)",
            "Building type",
            "Address",
        ]
        st.selectbox("Building label", _bl_opts, index=0, key="building_label")
        show_labels = st.checkbox("Show pipe labels", value=True, key="show_pipe_labels_cb")
        st.checkbox("Show building labels", value=True, key="show_building_labels")
        st.markdown("---")

        # ── Network controls ──
        st.subheader("Network")
        total_len_m = sum(p.get('length_m', 0.0) for p in st.session_state.pipe_network.pipes)
        n_connected = len(st.session_state.get('connected_buildings', {}))
        n_total     = len(st.session_state.b_gdf) if st.session_state.b_gdf is not None else 0
        st.markdown(f"""
<div style="background:#2d3348;border-radius:8px;padding:10px 14px;margin-bottom:8px">
  <div style="display:flex;justify-content:space-between;margin-bottom:6px">
    <span style="color:#8899bb;font-size:11px">PIPE LENGTH</span>
    <span style="color:#fff;font-weight:700;font-size:13px">{total_len_m:.0f} m</span>
  </div>
  <div style="display:flex;justify-content:space-between">
    <span style="color:#8899bb;font-size:11px">BUILDINGS CONNECTED</span>
    <span style="color:#27ae60;font-weight:700;font-size:13px">{n_connected} / {n_total}</span>
  </div>
</div>
""", unsafe_allow_html=True)

        sb_c1, sb_c2 = st.columns(2)
        with sb_c1:
            if st.button("🔄 Redraw", key="btn_redraw_sb"):
                st.session_state.map_rev += 1
                st.rerun()
        with sb_c2:
            if st.button("🗑️ Reset", key="btn_reset_sb"):
                st.session_state.pipe_network.pipes = []
                st.session_state.pipe_network.graph.clear()
                st.session_state.pipe_network.nodes = {}
                st.session_state.pipe_network.energy_hub = None
                st.session_state.connected_buildings = {}
                st.session_state.energy_hubs = {}
                st.session_state.flow_done = False
                st.session_state.map_rev += 1
                st.success("Network reset.")
                st.rerun()
        st.markdown("---")

        # ── Building Details ──
        st.subheader("Building Details")

        if st.session_state.b_gdf is not None and len(st.session_state.b_gdf) > 0:
            all_selected = st.session_state.get("select_all_buildings", False)
            btn_label = "✅ Deselect All Buildings" if all_selected else "☑️ Select All Buildings"
            if st.button(btn_label, key="btn_select_all_buildings", use_container_width=True):
                st.session_state.select_all_buildings = not all_selected
                st.session_state.selected_building = None
                st.rerun()

        if st.session_state.get("select_all_buildings", False):
            n_buildings = len(st.session_state.b_gdf)
            total_demand = st.session_state.b_gdf[
                'total_demand'].sum() if 'total_demand' in st.session_state.b_gdf.columns else 0
            st.markdown(f"""
        <div style="background:#2d3348;border-radius:8px;padding:12px 14px;margin-bottom:10px">
          <div style="font-size:11px;color:#8899bb;text-transform:uppercase;letter-spacing:.05em">All Buildings Selected</div>
          <div style="font-size:22px;font-weight:700;color:#27ae60">{n_buildings} <span style="font-size:13px;color:#8899bb">buildings</span></div>
          <div style="font-size:13px;color:#c8d0e0;margin-top:4px">Total demand: <b>{total_demand:.1f} MWh/a</b></div>
        </div>
        """, unsafe_allow_html=True)

            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("🔗 Connect All", use_container_width=True, key="btn_connect_all"):
                    if 'status' not in st.session_state.b_gdf.columns:
                        st.session_state.b_gdf['status'] = 'unconnected'
                    for idx in st.session_state.b_gdf.index:
                        if st.session_state.b_gdf.at[idx, 'status'] != 'hub':
                            st.session_state.b_gdf.at[idx, 'status'] = 'connected'
                    st.session_state.map_rev += 1
                    st.rerun()
            with col2:
                if st.button("✂️ Disconnect All", use_container_width=True, key="btn_disconnect_all"):
                    pn = st.session_state.pipe_network
                    cb = dict(st.session_state.get("connected_buildings", {}))
                    for bid, bnode in cb.items():
                        if bnode:
                            edges_to_remove = [
                                (u, v) for u, v in list(pn.graph.edges())
                                if u == bnode or v == bnode
                            ]
                            for u, v in edges_to_remove:
                                if pn.graph.has_edge(u, v):
                                    pn.graph.remove_edge(u, v)
                            pn.pipes = [
                                p for p in pn.pipes
                                if p["start_node"] != bnode and p["end_node"] != bnode
                            ]
                            pn.nodes.pop(bnode, None)
                            if pn.graph.has_node(bnode):
                                pn.graph.remove_node(bnode)
                    st.session_state.connected_buildings = {}
                    st.session_state.map_rev += 1
                    st.rerun()
            with col3:
                if st.button("🗑️ Delete All", use_container_width=True, key="btn_delete_all"):
                    if 'status' in st.session_state.b_gdf.columns:
                        hub_mask = st.session_state.b_gdf['status'] == 'hub'
                        st.session_state.b_gdf = st.session_state.b_gdf[hub_mask].copy()
                    else:
                        st.session_state.b_gdf = st.session_state.b_gdf.iloc[0:0].copy()
                    st.session_state.select_all_buildings = False
                    st.session_state.map_rev += 1
                    st.rerun()

        if st.session_state.b_gdf is not None and len(st.session_state.b_gdf) > 0:
            sel = st.session_state.get("selected_building", None)
            if sel is not None and sel in st.session_state.b_gdf.index:
                b = st.session_state.b_gdf.loc[sel]
                _sh   = float(b.get('space_heating', 0) or 0)
                _dhw  = float(b.get('domestic_hot_water', 0) or 0)
                _tot  = float(b.get('total_demand', 0) or 0) or (_sh + _dhw)
                _area = float(b.get('floor_area', 0) or 0)
                _sh_pct  = int(100 * _sh  / _tot) if _tot > 0 else 50
                _dhw_pct = 100 - _sh_pct

                st.markdown(f"""
<div style="background:#2d3348;border-radius:8px;padding:12px 14px;margin-bottom:10px">
  <div style="font-size:11px;color:#8899bb;text-transform:uppercase;letter-spacing:.05em">Address</div>
  <div style="font-size:13px;font-weight:600;color:#fff;margin-bottom:8px">{b.get('address','—')}</div>
  <div style="display:flex;gap:16px;margin-bottom:10px">
    <div>
      <div style="font-size:10px;color:#8899bb">TYPE</div>
      <div style="font-size:12px;color:#c8d0e0;font-weight:600">{b.get('building_type','Residential')}</div>
    </div>
    <div>
      <div style="font-size:10px;color:#8899bb">FLOOR AREA</div>
      <div style="font-size:12px;color:#c8d0e0;font-weight:600">{_area:.0f} m²</div>
    </div>
  </div>
  <div style="font-size:10px;color:#8899bb;text-transform:uppercase">Total heat demand</div>
  <div style="font-size:22px;font-weight:700;color:#27ae60">{_tot:.1f} <span style="font-size:13px;color:#8899bb">MWh/a</span></div>
  <div style="background:#1e2130;border-radius:4px;height:7px;margin:6px 0;overflow:hidden">
    <div style="height:100%;width:{_sh_pct}%;background:#e67e22;float:left"></div>
    <div style="height:100%;width:{_dhw_pct}%;background:#3498db;float:left"></div>
  </div>
  <div style="display:flex;gap:12px;font-size:11px;color:#8899bb">
    <span><span style="color:#e67e22">■</span> SH {_sh:.1f} MWh</span>
    <span><span style="color:#3498db">■</span> DHW {_dhw:.1f} MWh</span>
  </div>
</div>
""", unsafe_allow_html=True)

                with st.expander("✏️ Edit building data"):
                    new_type = st.selectbox(
                        "Type", list(DEMAND_PROFILES.keys()),
                        index=list(DEMAND_PROFILES.keys()).index(b.get('building_type', 'Residential'))
                    )
                    new_area = st.number_input("Floor area (m²)", min_value=0.0,
                        value=float(b.get('floor_area', 0) or 0.0), step=10.0)
                    new_sh  = st.number_input("Space heating (MWh/a)", min_value=0.0,
                        value=float(b.get('space_heating', 0) or 0.0), step=1.0)
                    new_dhw = st.number_input("DHW (MWh/a)", min_value=0.0,
                        value=float(b.get('domestic_hot_water', 0) or 0.0), step=1.0)
                    if st.button("Apply changes", key="apply_building_changes"):
                        st.session_state.b_gdf.at[sel, "building_type"]      = new_type
                        st.session_state.b_gdf.at[sel, "floor_area"]         = float(new_area)
                        st.session_state.b_gdf.at[sel, "space_heating"]      = float(new_sh)
                        st.session_state.b_gdf.at[sel, "domestic_hot_water"] = float(new_dhw)
                        st.session_state.b_gdf.at[sel, "total_demand"]       = float(new_sh) + float(new_dhw)
                        st.success("Updated.")
                        try:
                            if (not st.session_state.get("FAST_MODE")
                                    and st.session_state.pipe_network.energy_hub is not None):
                                start_flow_calc_async()
                        except Exception as e:
                            st.warning(f"Recalc skipped: {e}")

                is_connected = sel in st.session_state.connected_buildings
                c1, c2 = st.columns(2)
                with c1:
                    if not is_connected and st.button("🔗 Connect", key="btn_conn"):
                        try:
                            one = st.session_state.b_gdf.loc[[sel]]
                            n = st.session_state.pipe_network.auto_connect_buildings(one)
                            st.success(f"Connected {n} building.")
                            st.session_state.map_rev += 1
                            st.rerun()
                        except Exception as e:
                            st.error(str(e))
                with c2:
                    if is_connected and st.button("✂️ Disconnect", key="btn_disconn"):
                        bnode = st.session_state.connected_buildings.get(sel)
                        if bnode:
                            edges_to_remove = [
                                (u, v) for u, v in list(st.session_state.pipe_network.graph.edges())
                                if u == bnode or v == bnode
                            ]
                            for u, v in edges_to_remove:
                                if st.session_state.pipe_network.graph.has_edge(u, v):
                                    st.session_state.pipe_network.graph.remove_edge(u, v)
                            st.session_state.pipe_network.pipes = [
                                p for p in st.session_state.pipe_network.pipes
                                if p["start_node"] != bnode and p["end_node"] != bnode
                            ]
                            st.session_state.pipe_network.nodes.pop(bnode, None)
                            st.session_state.pipe_network.graph.remove_node(bnode)
                            st.session_state.connected_buildings.pop(sel, None)
                            if (not st.session_state.get("FAST_MODE")
                                    and st.session_state.pipe_network.energy_hub is not None):
                                start_flow_calc_async()
                            st.success("Disconnected.")
                            st.session_state.map_rev += 1
                            st.rerun()

                is_hub = sel in st.session_state.energy_hubs
                h1, h2 = st.columns(2)
                with h1:
                    if not is_hub and st.button("⚡ Set Hub", key="btn_set_hub"):
                        st.session_state.energy_hubs = {}
                        st.session_state.energy_hubs[sel] = {"source_type": "Boiler"}
                        c = b.geometry.centroid
                        src = st.session_state.pipe_network.add_node('source', (c.y, c.x), 0, sel)
                        st.session_state.pipe_network.energy_hub = src
                        st.success("Energy hub set.")
                        st.session_state.map_rev += 1
                        st.rerun()
                with h2:
                    if is_hub and st.button("✖ Unset Hub", key="btn_unset_hub"):
                        st.session_state.energy_hubs.pop(sel, None)
                        for nid, nd in list(st.session_state.pipe_network.nodes.items()):
                            if nd.get("type") == "source":
                                st.session_state.pipe_network.graph.remove_node(nid)
                                st.session_state.pipe_network.nodes.pop(nid, None)
                        st.session_state.pipe_network.energy_hub = None
                        st.info("Energy hub removed.")
                        st.session_state.map_rev += 1
                        st.rerun()

                st.markdown("---")
                if st.button("🗑️ Delete this building", key="btn_delete_building", use_container_width=True):
                    pn = st.session_state.pipe_network
                    # 1. Remove connection pipe + node if building was connected
                    bnode = st.session_state.connected_buildings.get(sel)
                    if bnode:
                        edges_to_rm = [(u, v) for u, v in list(pn.graph.edges())
                                       if u == bnode or v == bnode]
                        for u, v in edges_to_rm:
                            if pn.graph.has_edge(u, v):
                                pn.graph.remove_edge(u, v)
                        pn.pipes = [p for p in pn.pipes
                                    if p.get("start_node") != bnode and p.get("end_node") != bnode]
                        pn.nodes.pop(bnode, None)
                        try:
                            pn.graph.remove_node(bnode)
                        except Exception:
                            pass
                        st.session_state.connected_buildings.pop(sel, None)
                    # 2. Remove energy hub designation if this building was the hub
                    if is_hub:
                        st.session_state.energy_hubs.pop(sel, None)
                        for nid, nd in list(pn.nodes.items()):
                            if nd.get("type") == "source" and nd.get("building_id") == sel:
                                try:
                                    pn.graph.remove_node(nid)
                                except Exception:
                                    pass
                                pn.nodes.pop(nid, None)
                        pn.energy_hub = None
                    # 3. Remove building from GDF
                    st.session_state.b_gdf = st.session_state.b_gdf.drop(index=[sel])
                    st.session_state.selected_building = None
                    st.session_state.flow_done = False
                    st.session_state.map_rev += 1
                    st.success("Building deleted.")
                    st.rerun()

                with st.expander("📥 Data I/O"):
                    csv = st.session_state.b_gdf.to_csv(index=False).encode("utf-8")
                    st.download_button("⬇ Download buildings CSV", data=csv,
                                       file_name="buildings.csv", mime="text/csv")
                    up = st.file_uploader("Upload CSV", type=["csv"])
                    if up is not None:
                        try:
                            dfu = pd.read_csv(up)
                            if "address" in dfu.columns:
                                st.session_state.b_gdf = (
                                    st.session_state.b_gdf.set_index("address")
                                    .combine_first(dfu.set_index("address"))
                                    .reset_index()
                                )
                                st.success("CSV merged by address.")
                            else:
                                st.warning("CSV has no 'address' column.")
                        except Exception as e:
                            st.error(f"Upload failed: {e}")
            else:
                st.markdown('<div style="color:#8899bb;font-size:13px;padding:8px 0">Click a building on the map to see its details.</div>', unsafe_allow_html=True)
                if st.session_state.get("flow_running"):
                    st.info("Calculating flows…")
                elif st.session_state.get("flow_error"):
                    st.warning(f"Flow error: {st.session_state.flow_error}")
                elif st.session_state.get("flow_done"):
                    st.success("Flows updated ✓")
        else:
            st.markdown('<div style="color:#8899bb;font-size:13px;padding:8px 0">No buildings loaded yet. Select an area on the map to extract buildings.</div>', unsafe_allow_html=True)

        # ── Selected Pipe Panel ──────────────────────────────────────────────
        pidx = st.session_state.get("selected_pipe_idx")
        if pidx is not None and pidx < len(st.session_state.pipe_network.pipes):
            p = st.session_state.pipe_network.pipes[pidx]
            params = p.get("parameters", {}) or {}
            _p_len    = float(p.get('length_m', 0) or 0)
            _p_dn     = int(params.get('diameter', 0) or 0)
            _p_flow   = float(params.get('flow_rate', 0) or 0)
            _p_press  = float(params.get('pressure_loss_pa_per_m', 0) or 0)
            _p_hloss  = float(params.get('heat_loss', 0) or 0)
            _p_dheat  = float(params.get('delivered_heat', 0) or 0)
            _p_temp   = float(params.get('temp_out', 0) or 0)
            _p_tdrop  = float(params.get('temp_drop', 0) or 0)
            _p_type   = p.get('type', 'main').capitalize()
            _p_mat    = p.get('material', 'Steel')
            st.markdown("---")
            st.subheader("📐 Pipe Section")
            st.markdown(f"""
<div style="background:#1a2540;border-radius:8px;padding:12px 14px;font-size:13px;color:#c8d0e0;line-height:2">
  <div style="display:flex;justify-content:space-between;border-bottom:1px solid #2d3a55;padding-bottom:6px;margin-bottom:6px">
    <span style="color:#8899bb">Type</span><b style="color:#fff">{_p_type}</b>
  </div>
  <div style="display:flex;justify-content:space-between">
    <span style="color:#8899bb">Length</span><b>{_p_len:.1f} m</b>
  </div>
  <div style="display:flex;justify-content:space-between">
    <span style="color:#8899bb">Diameter</span><b>DN {_p_dn}</b>
  </div>
  <div style="display:flex;justify-content:space-between">
    <span style="color:#8899bb">Material</span><b>{_p_mat}</b>
  </div>
  <div style="border-top:1px solid #2d3a55;margin:6px 0"></div>
  <div style="display:flex;justify-content:space-between">
    <span style="color:#8899bb">Flow rate</span><b>{_p_flow:.2f} m³/day</b>
  </div>
  <div style="display:flex;justify-content:space-between">
    <span style="color:#8899bb">Pressure loss</span><b>{_p_press:.1f} Pa/m</b>
  </div>
  <div style="border-top:1px solid #2d3a55;margin:6px 0"></div>
  <div style="display:flex;justify-content:space-between">
    <span style="color:#8899bb">Delivered heat</span><b>{_p_dheat:.2f} MWh/a</b>
  </div>
  <div style="display:flex;justify-content:space-between">
    <span style="color:#8899bb">Heat loss</span><b>{_p_hloss:.2f} MWh/a</b>
  </div>
  <div style="display:flex;justify-content:space-between">
    <span style="color:#8899bb">Supply temp (out)</span><b>{_p_temp:.1f} °C</b>
  </div>
  <div style="display:flex;justify-content:space-between">
    <span style="color:#8899bb">Temp drop</span><b>{_p_tdrop:.2f} °C</b>
  </div>
</div>
""", unsafe_allow_html=True)
            _pc1, _pc2 = st.columns(2)
            with _pc1:
                if st.button("🗑️ Delete pipe", key="del_pipe_btn", use_container_width=True):
                    try:
                        pipe_to_del = st.session_state.pipe_network.pipes[pidx]
                        s, e = pipe_to_del.get("start_node"), pipe_to_del.get("end_node")
                        st.session_state.pipe_network.pipes.remove(pipe_to_del)
                        if s and e:
                            try:
                                st.session_state.pipe_network.graph.remove_edge(s, e)
                            except Exception:
                                pass
                        st.session_state.selected_pipe_idx = None
                        st.session_state.flow_done = False
                        st.session_state.map_rev += 1
                        st.rerun()
                    except Exception as ex:
                        st.error(f"Could not delete pipe: {ex}")
            with _pc2:
                if st.button("✖ Deselect", key="desel_pipe_btn", use_container_width=True):
                    st.session_state.selected_pipe_idx = None
                    st.rerun()

    # ── Map takes full remaining width ────────────────────────────────────
    # Map renders in full main area width

    def _building_label_text(row) -> str:
        """Format the building label according to the current label_select choice."""
        choice = st.session_state.get("building_label", "Total heat demand (MWh/a)")
        if choice == "Total heat demand (MWh/a)":
            return f"{float(row.get('total_demand', 0) or 0):.1f} MWh"
        if choice == "Space heating (MWh/a)":
            return f"{float(row.get('space_heating', 0) or 0):.1f} MWh"
        if choice == "Hot water demand (MWh/a)":
            return f"{float(row.get('domestic_hot_water', 0) or 0):.1f} MWh"
        if choice == "Floor area (m²)":
            return f"{float(row.get('floor_area', 0) or 0):.0f} m²"
        if choice == "Building type":
            return str(row.get("building_type", ""))
        if choice == "Address":
            return str(row.get("address", ""))
        return ""

    def _buildings_with_status(b_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Return a copy of b_gdf with a 'status' column: 'hub' | 'connected' | 'unconnected'.

        Derives connectivity from the network graph, making it robust to index changes.

        Args:
            b_gdf: Buildings GeoDataFrame.

        Returns:
            Copy with 'status' column added.
        """
        if b_gdf is None or b_gdf.empty:
            return b_gdf

        gdf = b_gdf.copy()
        gdf["status"] = "unconnected"

        connected_ids = set()
        hub_id = None
        try:
            pn = st.session_state.pipe_network
            G  = pn.graph
            hub_node = getattr(pn, "energy_hub", None)
            if hub_node is not None:
                hub_id = G.nodes[hub_node].get("building_id")
            for n, data in G.nodes(data=True):
                if data.get("type") == "building":
                    if G.degree(n) > 0:
                        bid = data.get("building_id")
                        if bid is not None:
                            connected_ids.add(str(bid))
        except Exception:
            pass

        if "building_id" in gdf.columns:
            gdf.loc[gdf["building_id"].astype(str).isin(connected_ids), "status"] = "connected"
            if hub_id is not None:
                gdf.loc[gdf["building_id"].astype(str) == str(hub_id), "status"] = "hub"
        else:
            conn_idx = set()
            try:
                conn_idx = set(st.session_state.get("connected_buildings", {}).keys())
            except Exception:
                pass
            gdf.loc[gdf.index.isin(conn_idx), "status"] = "connected"
            try:
                if hub_id is not None and hub_id in gdf.index:
                    gdf.loc[[hub_id], "status"] = "hub"
            except Exception:
                pass

        return gdf

    # ── Map rendering ────────────────────────────────────────────────────
    if 'map_center' not in st.session_state:
        st.session_state.map_center = [city_lat, city_lon]
    if 'map_zoom' not in st.session_state:
        st.session_state.map_zoom = 14

    # If buildings are loaded and we haven't locked the view to the project yet,
    # center on their bounding box so reruns stay on the work area.
    _b = st.session_state.get("b_gdf")
    if _b is not None and not _b.empty and not st.session_state.get("_map_locked_to_project"):
        try:
            _bounds = _b.total_bounds  # [minx, miny, maxx, maxy]
            st.session_state.map_center = [(_bounds[1] + _bounds[3]) / 2,
                                            (_bounds[0] + _bounds[2]) / 2]
            st.session_state.map_zoom   = 15
            st.session_state._map_locked_to_project = True
        except Exception:
            pass

    m = folium.Map(location=st.session_state.map_center, zoom_start=st.session_state.map_zoom)
    Draw(export=True).add_to(m)

    # Auto-activate the correct Draw toolbar button based on draw_mode
    _draw_mode_now = st.session_state.get("draw_mode", "Select Area")
    # Map draw_mode → Leaflet.draw CSS class on the toolbar button
    _btn_class = {
        "Select Area":   "leaflet-draw-draw-polygon",
        "Add Building":  "leaflet-draw-draw-rectangle",
        "Add Main Pipe": "leaflet-draw-draw-polyline",
    }.get(_draw_mode_now, "leaflet-draw-draw-polygon")
    _auto_draw_macro = MacroElement()
    _auto_draw_macro._template = Template(
        "{% macro html(this, kwargs) %}"
        "<script>"
        "(function autoActivateDraw() {"
        "  function tryClick() {"
        "    var btn = document.querySelector('." + _btn_class + "');"
        "    if (!btn) { setTimeout(tryClick, 400); return; }"
        "    // Only click if nothing is currently active"
        "    var active = document.querySelector('.leaflet-draw-toolbar a.leaflet-draw-toolbar-button-enabled');"
        "    if (!active) { btn.click(); }"
        "  }"
        "  if (document.readyState === 'loading') {"
        "    document.addEventListener('DOMContentLoaded', tryClick);"
        "  } else { setTimeout(tryClick, 600); }"
        "})();"
        "</script>"
        "{% endmacro %}"
    )
    _auto_draw_macro.add_to(m)

    MeasureControl().add_to(m)
    Fullscreen().add_to(m)
    MiniMap(toggle_display=True).add_to(m)
    MousePosition(prefix="Lat/Lon", separator=" , ", position="bottomright").add_to(m)

    # Buildings layer
    if st.session_state.b_gdf is not None and not st.session_state.b_gdf.empty:
        b_stat = st.session_state.b_gdf.copy()

        if 'is_energy_hub' not in b_stat.columns:
            b_stat['is_energy_hub'] = False

        cb = st.session_state.get("connected_buildings", {})
        connected_ids = set(cb.keys()) if isinstance(cb, dict) else set(cb)

        hub_ids = set()
        eh = st.session_state.get("energy_hubs", {})
        if isinstance(eh, dict):
            hub_ids |= set(eh.keys())

        b_stat["status"] = "unconnected"
        b_stat.loc[b_stat.index.isin(connected_ids), "status"] = "connected"
        hub_mask = (b_stat.get("is_energy_hub", False) == True) | (b_stat.index.isin(hub_ids))
        b_stat.loc[hub_mask, "status"] = "hub"

        tol = 0.00005 if (b_stat.crs and "4326" in str(b_stat.crs)) else 0.8
        gdf_for_map = simplify_for_map(b_stat, tol)

        show_building_labels = st.session_state.get("show_building_labels", True)
        _COLORS = {"unconnected": "#1e88e5", "connected": "#2e7d32", "hub": "#ff6f00"}

        _select_all = st.session_state.get("select_all_buildings", False)
        def _style(feat):
            s = feat["properties"].get("status", "unconnected")
            if _select_all and s != "hub":
                return {"fillColor": "#f1c40f", "color": "#f39c12", "weight": 2, "fillOpacity": 0.75}
            c = _COLORS.get(s, _COLORS["unconnected"])
            w = 3 if s == "hub" else 1
            fo = 0.85 if s == "hub" else 0.35
            border = "#b71c1c" if s == "hub" else c
            return {"fillColor": c, "color": border, "weight": w, "fillOpacity": fo}

        folium.GeoJson(
            gdf_for_map,
            name="Buildings",
            style_function=_style,
            tooltip=folium.GeoJsonTooltip(
                fields=["address", "total_demand", "building_type"],
                aliases=["Address:", "Demand (MWh/a):", "Type:"],
                localize=True,
                sticky=False,
            ),
        ).add_to(m)

        # Building labels — small DivIcon text at centroid
        if show_building_labels and not gdf_for_map.empty:
            for _, brow in gdf_for_map.iterrows():
                try:
                    centroid = brow.geometry.centroid
                    label_text = _building_label_text(brow)
                    if label_text:
                        folium.Marker(
                            location=[centroid.y, centroid.x],
                            icon=folium.DivIcon(
                                html=(
                                    f'<div style="font-size:10px;font-weight:700;'
                                    f'color:#111;text-shadow:1px 1px 0 #fff,-1px -1px 0 #fff,'
                                    f'1px -1px 0 #fff,-1px 1px 0 #fff;'
                                    f'white-space:nowrap;transform:translate(-50%,-50%);'
                                    f'pointer-events:none;">{label_text}</div>'
                                ),
                                icon_size=(1, 1),
                                icon_anchor=(0, 0),
                            ),
                        ).add_to(m)
                except Exception:
                    pass

        # Add a star marker on top of the energy hub building
        for idx in hub_ids:
            if idx in b_stat.index:
                try:
                    c = b_stat.loc[idx].geometry.centroid
                    folium.Marker(
                        location=[c.y, c.x],
                        icon=folium.DivIcon(
                            html='<div style="font-size:22px;text-align:center;'
                                 'line-height:22px;margin-top:-11px;margin-left:-11px;'
                                 'text-shadow:0 0 4px #fff,0 0 8px #fff;">⚡</div>',
                            icon_size=(22, 22),
                            icon_anchor=(11, 11),
                        ),
                        tooltip="Energy Hub",
                    ).add_to(m)
                except Exception:
                    pass

    # Pipes layer
    PIPE_VISUAL_WIDTH_PX_local = int(globals().get("PIPE_VISUAL_WIDTH_PX", 10))

    PIPE_STYLE_CHOICES_local = globals().get("PIPE_STYLE_CHOICES", {
        "Diameter (DN)": ("diameter", lambda v: f"DN{int(v)}" if v else "DN0"),
        "Pressure loss (Pa/m)": ("pressure_loss_pa_per_m", lambda v: f"{v:.0f} Pa/m"),
        "Heat losses (MWh/a)": ("heat_loss", lambda v: f"{v:.1f} MWh/a"),
        "Flow rate (m³/day)": ("flow_rate", lambda v: f"{v:.1f} m³/d"),
        "Temperature (°C)": ("temp_out", lambda v: f"{v:.1f} °C"),
    })

    pipe_metric_choice = st.session_state.get("pipe_metric", "Pressure loss (Pa/m)")
    metric_key, metric_fmt = PIPE_STYLE_CHOICES_local[pipe_metric_choice]

    pipes = st.session_state.pipe_network.pipes
    vals  = [float(p.get("parameters", {}).get(metric_key, 0.0) or 0.0) for p in pipes] or [0.0]
    vmin_, vmax_ = min(vals), max(vals)

    def _label_text(params: dict) -> str:
        """Format the pipe midpoint label for the active visualization metric."""
        if pipe_metric_choice == "Diameter (DN)":
            return f"DN{int(params.get('diameter', 0) or 0)}"
        if pipe_metric_choice == "Pressure loss (Pa/m)":
            return f"{float(params.get('pressure_loss_pa_per_m', 0) or 0):.0f} Pa/m"
        if pipe_metric_choice == "Heat loss (MWh/a)":
            return f"{float(params.get('heat_loss', 0) or 0):.1f} MWh/a"
        if pipe_metric_choice == "Flow rate (m³/day)":
            return f"{float(params.get('flow_rate', 0) or 0):.1f} m³/d"
        if pipe_metric_choice == "Temperature (°C)":
            return f"{float(params.get('temp_out', 0) or 0):.1f} °C"
        return ""

    for pipe_idx, p in enumerate(pipes):
        params = p.get("parameters", {})
        line   = shape(p["geometry"])
        val    = float(params.get(metric_key, 0.0) or 0.0)
        color  = _discrete_color(val, pipe_metric_choice)

        popup_html = f"""
        <div style="font-family: Arial; width: 260px" data-pipe-idx="{pipe_idx}">
          <h4 style="margin:0 0 6px 0;">Network section</h4>
          <table style="width:100%">
            <tr><td>Length</td><td>{p.get('length_m', 0):.1f} m</td></tr>
            <tr><td>Diameter</td><td>DN{params.get('diameter', 0)}</td></tr>
            <tr><td>Flow rate</td><td>{params.get('flow_rate', 0):.1f} m³/day</td></tr>
            <tr><td>Pressure loss</td><td>{params.get('pressure_loss_pa_per_m', 0):.0f} Pa/m</td></tr>
            <tr><td>Delivered heat</td><td>{params.get('delivered_heat', 0):.1f} MWh/a</td></tr>
            <tr><td>Heat losses</td><td>{params.get('heat_loss', 0):.1f} MWh/a</td></tr>
            <tr><td>Material</td><td>{p.get('material', 'Steel')}</td></tr>
            <tr><td>Supply temp (out)</td><td>{params.get('temp_out', 0):.1f} °C</td></tr>
            <tr><td>ΔT over segment</td><td>{params.get('temp_drop', 0):.2f} °C</td></tr>
          </table>
          <p style="font-size:11px;color:#888;margin:6px 0 0">Pipe #{pipe_idx} | Click sidebar to manage</p>
        </div>
        """

        coords = [(y, x) for x, y in list(line.coords)]
        weight = 14 if p.get("type") == "main" else 3

        folium.PolyLine(
            locations=coords,
            color=color,
            weight=weight,
            opacity=0.9,
            popup=folium.Popup(popup_html, max_width=320),
            tooltip=f"{pipe_metric_choice}: {metric_fmt(val)}",
        ).add_to(m)

        # Mid-point label — only shown when show_labels is True
        label_txt = _label_text(params)
        if label_txt and show_labels:
            mid = line.interpolate(line.length * 0.5)
            folium.Marker(
                location=[mid.y, mid.x],
                icon=folium.DivIcon(
                    html=f"""
                    <div style="
                        font: 700 13px/13px Arial, sans-serif;
                        color: #000;
                        background: transparent;
                        border: none;
                        padding: 0;
                        transform: translate(-50%,-50%);
                        white-space: nowrap;
                    ">{label_txt}</div>"""
                ),
            ).add_to(m)

    # Add legend inside the map as a Leaflet control (bottom-right corner)
    if pipes:
        _legend_elem = _make_discrete_legend(pipe_metric_choice, pipes)
        if _legend_elem is not None:
            _legend_elem.add_to(m)

    # Render the map — returned_objects limits what triggers reruns:
    # only fire on completed drawings and popup clicks, NOT on every mouse move / pan frame
    map_data = st_folium(
        m,
        height=700,
        width=None,
        key=f"main_map_{st.session_state.map_rev}",
        returned_objects=["all_drawings", "last_object_clicked", "last_object_clicked_popup"],
    )

    # Detect pipe click: try popup HTML first, then fall back to nearest-pipe geometry
    if map_data:
        popup_raw = map_data.get("last_object_clicked_popup")
        popup_str = str(popup_raw) if popup_raw else ""
        import re as _re
        _pipe_found = False
        if popup_str and "data-pipe-idx=" in popup_str:
            m_idx = _re.search(r'data-pipe-idx=["\'](\d+)["\']', popup_str)
            if m_idx:
                new_pipe_idx = int(m_idx.group(1))
                if new_pipe_idx != st.session_state.get("selected_pipe_idx"):
                    st.session_state.selected_pipe_idx = new_pipe_idx
                    st.rerun()
                _pipe_found = True

        # Fallback: if click coords are within ~15 m of any pipe, select it
        if not _pipe_found and map_data.get("last_object_clicked"):
            _clat = map_data["last_object_clicked"].get("lat")
            _clng = map_data["last_object_clicked"].get("lng")
            if _clat is not None and _clng is not None:
                _click_pt = Point(_clng, _clat)
                _best_idx = None
                _best_d   = 0.00015  # ~15 m in degrees
                for _pi, _pp in enumerate(st.session_state.pipe_network.pipes):
                    try:
                        _pl = shape(_pp["geometry"])
                        _d  = _pl.distance(_click_pt)
                        if _d < _best_d:
                            _best_d   = _d
                            _best_idx = _pi
                    except Exception:
                        continue
                if _best_idx is not None and _best_idx != st.session_state.get("selected_pipe_idx"):
                    st.session_state.selected_pipe_idx = _best_idx
                    st.rerun()

    # Click to select nearest building
    if map_data and map_data.get("last_object_clicked") and st.session_state.b_gdf is not None:
        click_lat = map_data["last_object_clicked"]["lat"]
        click_lng = map_data["last_object_clicked"]["lng"]
        click_pt  = Point(click_lng, click_lat)

        gdf = st.session_state.b_gdf
        centroids = gdf.geometry.centroid
        dists = centroids.distance(click_pt)
        if len(dists) > 0:
            i_min = dists.idxmin()
            try:
                epsg = 3857
                d = gpd.GeoSeries([click_pt], crs="EPSG:4326").to_crs(epsg=epsg).iloc[0].distance(
                    gpd.GeoSeries([centroids.loc[i_min]], crs="EPSG:4326").to_crs(epsg=epsg).iloc[0]
                )
                if d <= 50 and i_min != st.session_state.get("selected_building"):
                    st.session_state.selected_building = i_min
            except Exception:
                if i_min != st.session_state.get("selected_building"):
                    st.session_state.selected_building = i_min

    # Drawing handler — use only all_drawings (completed shapes), never last_active_drawing
    # last_active_drawing fires on every click mid-draw and causes mid-draw reruns
    drawing = None
    if map_data:
        if isinstance(map_data.get("all_drawings"), list) and map_data["all_drawings"]:
            drawing = map_data["all_drawings"][-1]
        # De-duplicate: skip if this exact geometry was already processed last rerun
        if drawing is not None:
            import json as _json
            _drawing_sig = _json.dumps(drawing.get("geometry", {}), sort_keys=True)
            if _drawing_sig == st.session_state.get("_last_drawing_sig"):
                drawing = None  # already handled
            else:
                st.session_state._last_drawing_sig = _drawing_sig

    if drawing:

        if drawing["geometry"]["type"] == "Polygon":
            if st.session_state.draw_mode == "Select Area":
                st.session_state.selected_areas.append(drawing)
                _status = st.empty()
                _status.info("⏳ Contacting OpenStreetMap… this takes 10–30 s. Please wait.")
                try:
                    import uuid as _uuid
                    poly = shape(drawing["geometry"])

                    # Area guard: warn if polygon is very large (> ~4 km²)
                    try:
                        _c = poly.centroid
                        _utm = (32600 if _c.y >= 0 else 32700) + int((_c.x + 180) // 6) + 1
                        _poly_m2 = gpd.GeoSeries([poly], crs="EPSG:4326").to_crs(epsg=_utm).area.iloc[0]
                        if _poly_m2 > 4_000_000:
                            _status.empty()
                            st.warning(
                                f"⚠️ Selected area is very large ({_poly_m2/1e6:.1f} km²). "
                                "OSM may return thousands of buildings or time out. "
                                "Consider drawing a smaller area."
                            )
                            st.stop()
                    except Exception:
                        pass

                    try:
                        b = fetch_osm_buildings(drawing["geometry"])
                    except RuntimeError as _e:
                        _status.empty()
                        st.error(f"⚠️ {_e}")
                        st.stop()
                    _status.empty()

                    if b.empty:
                        st.warning("⚠️ No buildings found. Try a different area or check the OSM coverage.")
                    else:
                        def _addr(r):
                            s = r.get("addr:street", "")
                            n = r.get("addr:housenumber", "")
                            a = (f"{s} {n}").strip()
                            return a if a else "OSM building"

                        b["address"] = b.apply(_addr, axis=1)
                        b = assign_missing_addresses(b, prefix="Building")

                        c = poly.centroid
                        utm_zone = int((c.x + 180) // 6) + 1
                        epsg = (32600 if c.y >= 0 else 32700) + utm_zone
                        footprint_m2 = b.to_crs(epsg=epsg).area

                        # Exclude unheated outbuildings (garages, sheds, carports, ...)
                        _btag = (b["building"].astype(str).str.lower()
                                 if "building" in b.columns else pd.Series("yes", index=b.index))
                        _has_nr = (b["addr:housenumber"].notna()
                                   if "addr:housenumber" in b.columns else pd.Series(False, index=b.index))
                        _unheated = _btag.isin(UNHEATED_BUILDING_TAGS) | (
                            (_btag == "yes") & (footprint_m2 < MIN_HEATED_FOOTPRINT_M2) & ~_has_nr
                        )
                        _n_excl = int(_unheated.sum())
                        b = b[~_unheated].copy()
                        footprint_m2 = footprint_m2[~_unheated]
                        if _n_excl:
                            st.info(f"{_n_excl} outbuildings (garages, sheds, carports, ...) were not imported.")
                        if b.empty:
                            st.warning("⚠️ Only outbuildings found in this area.")
                            st.stop()

                        levels_series = pd.Series(np.nan, index=b.index, dtype="float64")
                        for col in ("building:levels", "levels"):
                            if col in b.columns:
                                levels_series = levels_series.fillna(
                                    pd.to_numeric(b[col], errors="coerce")
                                )

                        height = (pd.to_numeric(b.get("height"), errors="coerce")
                                  if "height" in b.columns else np.nan)
                        min_height = (pd.to_numeric(b.get("min_height"), errors="coerce")
                                      if "min_height" in b.columns else 0.0)
                        if isinstance(height, pd.Series):
                            est = ((height.fillna(np.nan) - (
                                min_height if isinstance(min_height, pd.Series) else 0.0
                            )) / 3.0).round()
                            levels_series = levels_series.fillna(est)

                        if "building:levels:underground" in b.columns:
                            ug = pd.to_numeric(
                                b["building:levels:underground"], errors="coerce"
                            ).fillna(0.0)
                            levels_series = (levels_series - ug).clip(lower=1)

                        levels_series = levels_series.clip(lower=1, upper=8).fillna(2.0)
                        b["floor_area"] = (footprint_m2 * levels_series * 0.70).astype(float)

                        # Auto-detect building type from OSM 'building' tag
                        _OSM_TYPE_MAP = {
                            # Residential
                            "apartments": "Residential", "residential": "Residential",
                            "house": "Residential", "detached": "Residential",
                            "semidetached_house": "Residential", "terrace": "Residential",
                            "dormitory": "Residential", "bungalow": "Residential",
                            "farm": "Residential", "static_caravan": "Residential",
                            # Office / commercial
                            "office": "Office", "commercial": "Office",
                            "retail": "Office", "supermarket": "Office",
                            "hotel": "Office", "civic": "Office",
                            # School / education
                            "school": "School", "university": "School",
                            "college": "School", "kindergarten": "School",
                            "library": "School",
                            # Hospital / health
                            "hospital": "Hospital", "clinic": "Hospital",
                            "healthcare": "Hospital",
                            # Industrial
                            "industrial": "Industrial", "warehouse": "Industrial",
                            "factory": "Industrial", "manufacture": "Industrial",
                        }
                        # Use the 'building' tag; where it is only 'yes', fall back to
                        # 'building:use', 'amenity' or 'shop' of the same building.
                        _tag = (b["building"].astype(str).str.lower()
                                if "building" in b.columns else pd.Series("yes", index=b.index))
                        for _tc in ("building:use", "amenity", "shop"):
                            if _tc in b.columns:
                                _alt = b[_tc].astype(str).str.lower()
                                _tag = _tag.where((_tag != "yes") | (_alt.isin(["nan", "none", ""])), _alt)
                        if "shop" in b.columns:
                            _OSM_TYPE_MAP.setdefault("shop", "Office")
                        b["building_type"] = _tag.map(_OSM_TYPE_MAP).fillna("Residential")

                        b["space_heating"]      = 0.0
                        b["domestic_hot_water"] = 0.0
                        b["total_demand"]       = 0.0

                        def _auto_fill_demand(row):
                            d = calculate_heat_demand(row["floor_area"], row["building_type"])
                            row["space_heating"]      = d["space_heating"]
                            row["domestic_hot_water"] = d["dhw"]
                            row["total_demand"]       = d["total"]
                            return row

                        b = b.apply(_auto_fill_demand, axis=1)

                        if "building_id" not in b.columns:
                            if "osmid" in b.columns:
                                b["building_id"] = b["osmid"].astype(str)
                            else:
                                b["building_id"] = [f"osm_{_uuid.uuid4().hex[:8]}" for _ in range(len(b))]

                        st.session_state.b_gdf = (
                            b if st.session_state.b_gdf is None
                            else pd.concat([st.session_state.b_gdf, b], ignore_index=True)
                        )
                        try:
                            if "metric_crs" not in st.session_state or st.session_state.get("metric_crs") is None:
                                st.session_state.metric_crs = b.estimate_utm_crs()
                            # Always reproject the FULL merged GDF so distance calcs work for all buildings
                            st.session_state.b_gdf_m = st.session_state.b_gdf.to_crs(st.session_state.metric_crs)
                        except Exception:
                            st.session_state.metric_crs = None
                            st.session_state.b_gdf_m    = None

                        st.session_state.selected_areas.clear()
                        st.session_state.map_rev += 1
                        st.rerun()

                except Exception as e:
                    _status.empty()
                    st.error(f"❌ Could not extract buildings: {e}\n\nTry drawing a smaller area or wait a moment and try again.")
                    st.rerun()

            elif st.session_state.draw_mode == "Add Building":
                building_name = f"Manually Added Building {st.session_state.manual_building_counter}"
                st.session_state.manual_building_counter += 1

                geom     = shape(drawing["geometry"])
                area_m2  = calculate_building_area(geom)

                # nPro-style calibration: Area / Demand ≈ 8
                total_MWh = max(area_m2 / 8.0, 0.0)
                sh_share  = 74.0 / (74.0 + 21.0)
                dhw_share = 1.0 - sh_share
                sh_MWh    = total_MWh * sh_share
                dhw_MWh   = total_MWh * dhw_share

                new_b = gpd.GeoDataFrame([{
                    "geometry":          geom,
                    "floor_area":        float(area_m2),
                    "address":           building_name,
                    "building_type":     "Residential",
                    "space_heating":     float(sh_MWh),
                    "domestic_hot_water":float(dhw_MWh),
                    "total_demand":      float(total_MWh),
                }], geometry="geometry", crs="EPSG:4326")

                st.session_state.b_gdf = (
                    new_b if st.session_state.b_gdf is None
                    else pd.concat([st.session_state.b_gdf, new_b], ignore_index=True)
                )

                try:
                    if "metric_crs" not in st.session_state or st.session_state.metric_crs is None:
                        st.session_state.metric_crs = new_b.estimate_utm_crs()
                    st.session_state.b_gdf_m = (
                        st.session_state.b_gdf.to_crs(st.session_state.metric_crs)
                        if st.session_state.metric_crs else None
                    )
                except Exception:
                    pass

                st.success(f"Added building '{building_name}' ({area_m2:.1f} m²) with {total_MWh:.1f} MWh/a.")
                st.session_state.map_rev += 1
                d = calculate_heat_demand(area_m2, "Residential")
                new_b["space_heating"]     = d["space_heating"]
                new_b["domestic_hot_water"]= d["dhw"]
                new_b["total_demand"]      = d["total"]
                st.rerun()

        elif drawing["geometry"]["type"] == "LineString":
            if st.session_state.draw_mode == "Add Main Pipe":
                try:
                    if not drawing or not drawing.get("geometry") or not drawing["geometry"].get("coordinates"):
                        st.error("Invalid line drawing - please try again")
                        st.stop()

                    line_geom = shape(drawing["geometry"])
                    coords    = list(line_geom.coords)

                    if len(coords) < 2:
                        st.error("Line must have at least 2 points")
                        st.stop()

                    SNAP_THRESHOLD_DEG = 0.0002  # ~22 m
                    SNAP_THRESHOLD_M   = 25.0    # metres — for mid-pipe snap

                    def _snap_or_create(pn, lat, lon):
                        """Return an existing node ID if within snap threshold.
                        If no node is close but the point lands on the middle of an
                        existing main pipe, split that pipe and return the new junction.
                        Otherwise create a new disconnected junction."""
                        # 1. Try snapping to an existing node first
                        best_id = None
                        best_d  = SNAP_THRESHOLD_DEG
                        for nid, nd in pn.nodes.items():
                            pos = nd.get('pos', (0, 0))
                            d   = ((pos[0] - lat) ** 2 + (pos[1] - lon) ** 2) ** 0.5
                            if d < best_d:
                                best_d  = d
                                best_id = nid
                        if best_id is not None:
                            return best_id

                        # 2. Check if point lies on (or near) the middle of any main pipe
                        pt_geo = Point(lon, lat)
                        proj_epsg = None
                        try:
                            mc = st.session_state.get("metric_crs")
                            if mc is not None:
                                proj_epsg = int(mc.to_epsg() or 3857)
                        except Exception:
                            pass

                        best_pipe  = None
                        best_dist  = SNAP_THRESHOLD_M
                        best_conn  = None
                        for p in list(pn.pipes):
                            if p.get('type') != 'main':
                                continue
                            pipe_line = shape(p['geometry'])
                            try:
                                if proj_epsg is not None:
                                    pl_m = gpd.GeoSeries([pipe_line], crs="EPSG:4326").to_crs(epsg=proj_epsg).iloc[0]
                                    pt_m = gpd.GeoSeries([pt_geo],   crs="EPSG:4326").to_crs(epsg=proj_epsg).iloc[0]
                                    d_m  = float(pl_m.distance(pt_m))
                                    if d_m < best_dist:
                                        best_dist = d_m
                                        best_pipe = p
                                        # projected snap point → back to 4326
                                        snap_m = pl_m.interpolate(pl_m.project(pt_m))
                                        best_conn = gpd.GeoSeries([snap_m], crs=f"EPSG:{proj_epsg}").to_crs(epsg=4326).iloc[0]
                                else:
                                    d_deg = float(pipe_line.distance(pt_geo))
                                    if d_deg < best_dist / 111320.0:
                                        best_dist = d_deg * 111320.0
                                        best_pipe = p
                                        best_conn = pipe_line.interpolate(pipe_line.project(pt_geo))
                            except Exception:
                                continue

                        if best_pipe is not None and best_conn is not None:
                            # Split the existing main pipe at the snap point
                            pipe_line = shape(best_pipe['geometry'])
                            coords_p  = list(pipe_line.coords)
                            cx, cy    = best_conn.x, best_conn.y

                            # Find insert position
                            cum = [0.0]
                            for i in range(len(coords_p) - 1):
                                cum.append(cum[-1] + LineString([coords_p[i], coords_p[i+1]]).length)
                            s_param  = pipe_line.project(Point(cx, cy))
                            ins_idx  = None
                            for i in range(len(cum) - 1):
                                if cum[i] - 1e-9 <= s_param <= cum[i+1] + 1e-9:
                                    ins_idx = i
                                    break

                            if ins_idx is not None:
                                new_coords = coords_p[:ins_idx+1] + [(cx, cy)] + coords_p[ins_idx+1:]
                                seg1 = LineString(new_coords[:ins_idx+2])
                                seg2 = LineString(new_coords[ins_idx+1:])
                                if seg1.length > 1e-9 and seg2.length > 1e-9:
                                    junc_id = pn.add_node('junction', (cy, cx))
                                    pn.add_pipe_segment(best_pipe['start_node'], junc_id,
                                                        mapping(seg1), best_pipe.get('material'), 'main')
                                    pn.add_pipe_segment(junc_id, best_pipe['end_node'],
                                                        mapping(seg2), best_pipe.get('material'), 'main')
                                    if pn.graph.has_edge(best_pipe['start_node'], best_pipe['end_node']):
                                        pn.graph.remove_edge(best_pipe['start_node'], best_pipe['end_node'])
                                    try:
                                        pn.pipes.remove(best_pipe)
                                    except ValueError:
                                        if 'id' in best_pipe:
                                            pn.pipes[:] = [pp for pp in pn.pipes if pp.get('id') != best_pipe['id']]
                                    return junc_id

                        # 3. Fallback: create a new isolated junction
                        return pn.add_node('junction', (lat, lon))

                    start_node = _snap_or_create(
                        st.session_state.pipe_network, coords[0][1], coords[0][0]
                    )
                    end_node = _snap_or_create(
                        st.session_state.pipe_network, coords[-1][1], coords[-1][0]
                    )

                    # Dedup guard: skip if this exact drawing was already processed
                    _draw_sig = str(drawing["geometry"]["coordinates"])
                    if st.session_state.get("_last_pipe_sig") == _draw_sig:
                        st.stop()
                    st.session_state["_last_pipe_sig"] = _draw_sig

                    pipe_id = st.session_state.pipe_network.add_polyline_pipe(
                        start_node=start_node,
                        end_node=end_node,
                        geometry=mapping(line_geom),
                        material=SYSTEM_PARAMS['default_material'],
                        pipe_type='main'
                    )

                    st.success("✅ Pipe segment added! Draw another segment or switch mode when done.")
                    # Increment map_rev so the map widget gets a fresh key (clears all_drawings),
                    # then rerun — the auto-draw macro will re-activate the polyline tool so
                    # the user can immediately draw the next segment.
                    st.session_state.map_rev += 1
                    st.rerun()

                except Exception as e:
                    st.error(f"Error processing pipe drawing: {e}")
                    st.stop()


    # Network Analysis Section

    with st.expander("Network Analysis"):
        if st.session_state.pipe_network.pipes:
            main_pipes       = [p for p in st.session_state.pipe_network.pipes if p.get('type') == 'main']
            connection_pipes = [p for p in st.session_state.pipe_network.pipes if p.get('type') == 'connection']

            total_length        = sum(pipe.get('length_m', 0) for pipe in st.session_state.pipe_network.pipes)
            total_pressure_loss = sum(
                pipe.get('parameters', {}).get('pressure_loss', 0)
                for pipe in st.session_state.pipe_network.pipes
            )
            total_heat_loss     = sum(
                pipe.get('parameters', {}).get('heat_loss', 0)
                for pipe in st.session_state.pipe_network.pipes
            )
            # Total Delivered Heat: use only main pipes to avoid double-counting
            # and inflation from manually-added buildings on connection pipes.
            main_pipe_delivered = sum(
                pipe.get('parameters', {}).get('delivered_heat', 0)
                for pipe in main_pipes
            )
            conn_pipe_delivered = sum(
                pipe.get('parameters', {}).get('delivered_heat', 0)
                for pipe in connection_pipes
            )
            # Use the larger of the two pipe groups' delivered heat (main pipes
            # carry cumulative flow; connection pipes carry per-building flow).
            # When simulation has run, main pipes give the most reliable total.
            # The pipe at the energy hub carries the heat of all buildings -> largest value = total
            total_delivered_heat = max(
                (float(p.get('parameters', {}).get('delivered_heat', 0) or 0)
                 for p in st.session_state.pipe_network.pipes),
                default=0.0)

            main_len = sum(p.get('length_m', 0) for p in main_pipes)
            conn_len = sum(p.get('length_m', 0) for p in connection_pipes)

            # Row 1: pipe lengths
            r1c1, r1c2, r1c3 = st.columns(3)
            r1c1.metric("Total Pipe Length", f"{total_length:.0f} m")
            r1c2.metric("Main Pipes", f"{main_len:.0f} m")
            r1c3.metric("Connection Pipes", f"{conn_len:.0f} m")

            # Compute design peak, pump power, efficiency
            design_peak_kW = 0.0
            pump_power_kW  = 0.0
            efficiency     = 0.0
            n_conn         = 0

            if st.session_state.b_gdf is not None and st.session_state.pipe_network.nodes:
                pn = st.session_state.pipe_network
                b_gdf = st.session_state.b_gdf
                connected_bids = {
                    ndata.get('building_id')
                    for nid, ndata in pn.nodes.items()
                    if ndata.get('type') == 'building' and ndata.get('building_id') is not None
                }
                conn_gdf = b_gdf[b_gdf.index.isin(connected_bids)] if connected_bids else b_gdf

                # Exclude manually-added buildings from peak load estimation —
                # they have no real OSM floor area so their demands skew results.
                osm_mask = ~conn_gdf.get('address', pd.Series(dtype=str)).astype(str).str.startswith(
                    'Manually Added Building'
                )
                osm_conn_gdf = conn_gdf[osm_mask]
                n_conn = len(osm_conn_gdf)

                if n_conn > 0:
                    total_annual_MWh = float(osm_conn_gdf['total_demand'].sum())
                    avg_kW = (total_annual_MWh * 1000.0) / 8760.0
                    peak_to_avg = float(DEMAND_PROFILES.get('Residential', {}).get('peak_to_average', 3.7))

                    # Winter (2001) diversity factor formula
                    _A, _B, _C, _D = 0.449677646267461, 0.551234688, 53.84382392, 1.762743268
                    n_c = min(n_conn, 200)
                    div_factor = 1.0 if n_conn <= 1 else _A + _B / (1.0 + (n_c / _C) ** _D)
                    design_peak_kW = avg_kW * peak_to_avg * div_factor

            if total_pressure_loss > 0 and st.session_state.pipe_network.pipes:
                pn = st.session_state.pipe_network
                # Find hub pipe (pipe leaving the energy hub source node with maximum flow)
                hub_node = getattr(pn, 'energy_hub', None)
                hub_pipe = None
                max_flow = -1.0
                for p in pn.pipes:
                    if hub_node and p.get('start_node') == hub_node:
                        q = float(p.get('parameters', {}).get('flow_rate', 0.0))
                        if q > max_flow:
                            max_flow = q
                            hub_pipe = p
                if hub_pipe is None:
                    hub_pipe = max(
                        pn.pipes,
                        key=lambda p: float(p.get('parameters', {}).get('flow_rate', 0.0)),
                        default=None
                    )
                if hub_pipe is not None:
                    q_m3s = float(hub_pipe['parameters'].get('flow_rate', 0.0)) / 86400.0
                    pump_power_kW = (q_m3s * total_pressure_loss) / (
                        1000.0 * SYSTEM_PARAMS['pump_efficiency']
                    )
                    # Compute actual peak thermal power from hub pipe flow:
                    # Q_kW = m_dot [kg/s] × cp [kJ/(kg·K)] × ΔT [K]
                    # m_dot = flow_m3s × 1000 kg/m³ (water density)
                    # cp = 4.186 kJ/(kg·K), ΔT = supply − return = 20 K (default)
                    delta_T = float(SYSTEM_PARAMS.get('supply_temp', 80.0)) - float(SYSTEM_PARAMS.get('return_temp', 60.0))
                    design_peak_kW = q_m3s * 1000.0 * 4.186 * delta_T

            if total_delivered_heat > 0:
                efficiency = (total_delivered_heat / (total_delivered_heat + total_heat_loss)) * 100

            # Connected buildings count
            n_buildings_in_network = len(st.session_state.get('connected_buildings', {}))

            flow_done = st.session_state.get('flow_done', False)

            # ── nPro-style Results panel ─────────────────────────────────
            st.markdown('<div class="section-label">📊 Results</div>', unsafe_allow_html=True)
            res_tab1, res_tab2, res_tab3 = st.tabs(["🔥 Summary", "🔧 Network", "📐 DN Distribution"])

            with res_tab1:
                # Gather building-level demand totals (OSM only)
                sh_total_MWh  = 0.0
                dhw_total_MWh = 0.0
                if st.session_state.b_gdf is not None:
                    _bgdf = st.session_state.b_gdf
                    _osm  = _bgdf[~_bgdf.get('address', pd.Series(dtype=str)).astype(str)
                                  .str.startswith('Manually Added Building')]
                    sh_total_MWh  = float(_osm.get('space_heating',
                                          pd.Series(dtype=float)).sum())
                    dhw_total_MWh = float(_osm.get('domestic_hot_water',
                                          pd.Series(dtype=float)).sum())

                demand_total_MWh = sh_total_MWh + dhw_total_MWh
                # Heat feed-in = building demand + network heat losses
                heat_feed_in_MWh = (demand_total_MWh + total_heat_loss) if demand_total_MWh > 0 else 0.0
                peak_feed_in_kW  = (design_peak_kW + total_heat_loss * 1000.0
                                    / float(SYSTEM_PARAMS.get('operating_hours', 8760))) if design_peak_kW > 0 else 0.0

                # nPro-style heat results table
                _peak_str = '—' if design_peak_kW == 0 else f'{design_peak_kW:.0f} kW'
                _losses_str = '—' if total_heat_loss == 0 else f'{total_heat_loss:.1f} MWh'
                _pump_str = '—' if pump_power_kW == 0 else f'{pump_power_kW:.2f} kW'
                _feed_e_str = '—' if heat_feed_in_MWh == 0 else f'{heat_feed_in_MWh:.1f} MWh'
                _feed_p_str = '—' if peak_feed_in_kW == 0 else f'{peak_feed_in_kW:.0f} kW'
                st.markdown(f"""
<table class="res-table">
  <thead>
    <tr><th></th><th>Annual energy</th><th>Peak power</th></tr>
  </thead>
  <tbody>
    <tr><td>Space heating</td><td>{sh_total_MWh:.0f} MWh</td><td>—</td></tr>
    <tr><td>Domestic hot water</td><td>{dhw_total_MWh:.0f} MWh</td><td>—</td></tr>
    <tr class="total-row"><td>Total demand</td><td>{demand_total_MWh:.0f} MWh</td><td>{_peak_str}</td></tr>
    <tr class="section-row"><td colspan="3">Heat network</td></tr>
    <tr><td>Heat losses</td><td>{_losses_str}</td><td>—</td></tr>
    <tr class="total-row"><td>Heat feed-in at energy hub</td><td>{_feed_e_str}</td><td>{_feed_p_str}</td></tr>
  </tbody>
</table>
""", unsafe_allow_html=True)

                st.markdown(f"""
<div style="margin-top:12px;font-size:12px;color:#778;">
  Buildings connected: <b>{n_buildings_in_network}</b> &nbsp;|&nbsp;
  System efficiency: <b>{'—' if efficiency == 0 else f'{efficiency:.1f} %'}</b>
</div>
""", unsafe_allow_html=True)
                _hi = st.session_state.get('heating_peak_info')
                if _hi:
                    st.caption(f"Space-heating peak from TMY degree-hour profile ({_hi['location']}): "
                               f"T_base {_hi['t_base']:.0f} °C, design/coldest temperature {_hi['t_min']:.1f} °C, "
                               f"full-load hours {_hi['flh']:.0f} h/a.")
                elif HEATING_PEAK.get('method') == 'profile':
                    st.caption("⚠️ Weather data could not be loaded — space-heating peak from the "
                               "peak-to-average factor (3.7 × 1.2) instead.")
                else:
                    st.caption("Space-heating peak from peak-to-average factor.")

            with res_tab2:
                # Pipe lengths table
                st.markdown(f"""
<table class="res-table">
  <thead>
    <tr><th>Category</th><th>Length</th><th>Count</th></tr>
  </thead>
  <tbody>
    <tr><td>Main pipes</td><td>{main_len:.0f} m</td><td>{len(main_pipes)}</td></tr>
    <tr><td>Connection pipes</td><td>{conn_len:.0f} m</td><td>{len(connection_pipes)}</td></tr>
    <tr class="total-row"><td>Total</td><td>{total_length:.0f} m ({total_length/1000:.2f} km)</td><td>{len(st.session_state.pipe_network.pipes)}</td></tr>
  </tbody>
</table>
""", unsafe_allow_html=True)

                if flow_done:
                    st.markdown("<div style='margin-top:10px'></div>", unsafe_allow_html=True)
                    _eff_str2 = f'{efficiency:.1f} %'
                    _peak2_str = f'{design_peak_kW:.1f} kW'
                    _pump2_str = f'{pump_power_kW:.2f} kW'
                    _loss2_str = f'{total_heat_loss:.1f} MWh/a'
                    _deliv_str = f'{total_delivered_heat:.1f} MWh/a'
                    st.markdown(f"""
<table class="res-table">
  <thead>
    <tr><th>Metric</th><th>Value</th></tr>
  </thead>
  <tbody>
    <tr><td>Design peak load</td><td>{_peak2_str}</td></tr>
    <tr><td>Total heat losses</td><td>{_loss2_str}</td></tr>
    <tr><td>Total delivered heat</td><td>{_deliv_str}</td></tr>
    <tr class="total-row"><td>System efficiency</td><td>{_eff_str2}</td></tr>
  </tbody>
</table>
""", unsafe_allow_html=True)
                else:
                    st.info("Run simulation (▶ Recalculate Flows) to see hydraulic results.")

            with res_tab3:
                from collections import Counter
                dn_counts  = Counter()
                dn_lengths = {}
                dn_main    = {}
                dn_conn    = {}
                for p in st.session_state.pipe_network.pipes:
                    params  = p.get('parameters', {})
                    dn      = params.get('diameter')
                    ptype   = p.get('type', '')
                    if dn is not None:
                        dn_key = f"DN {int(dn)}"
                        dn_counts[dn_key]  += 1
                        dn_lengths[dn_key]  = dn_lengths.get(dn_key, 0.0) + p.get('length_m', 0.0)
                        if ptype == 'main':
                            dn_main[dn_key] = dn_main.get(dn_key, 0.0) + p.get('length_m', 0.0)
                        else:
                            dn_conn[dn_key] = dn_conn.get(dn_key, 0.0) + p.get('length_m', 0.0)

                if dn_counts:
                    _sorted_dn = sorted(dn_counts, key=lambda x: int(x.split()[1]))
                    rows = "".join(
                        f"<tr><td>{dk}</td><td>{dn_counts[dk]}</td>"
                        f"<td>{dn_lengths[dk]:.0f} m</td>"
                        f"<td>{dn_main.get(dk, 0):.0f} m</td>"
                        f"<td>{dn_conn.get(dk, 0):.0f} m</td></tr>"
                        for dk in _sorted_dn
                    )
                    st.markdown(
                        f'<table class="res-table"><thead>'
                        f'<tr><th>DN</th><th>Pipes</th><th>Total length</th><th>Main</th><th>Connection</th></tr>'
                        f'</thead><tbody>{rows}</tbody></table>',
                        unsafe_allow_html=True,
                    )

                    # Bar chart using st.bar_chart (always available, no altair dependency)
                    _chart_df = pd.DataFrame(
                        {"Main (m)": [dn_main.get(k, 0) for k in _sorted_dn],
                         "Connection (m)": [dn_conn.get(k, 0) for k in _sorted_dn]},
                        index=_sorted_dn,
                    )
                    st.markdown("<div style='margin-top:10px;font-size:12px;color:#556;font-weight:600'>Pipe length by DN class</div>",
                                unsafe_allow_html=True)
                    st.bar_chart(_chart_df, use_container_width=True, height=220, color=["#1e3a5f", "#27ae60"])
                else:
                    st.info("No pipes in the network yet.")
