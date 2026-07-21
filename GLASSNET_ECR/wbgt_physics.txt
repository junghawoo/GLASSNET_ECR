"""
Pure-Python teaching implementation of the explicit Liljegren WBGT model.

Purpose
-------
This module lets students inspect and run the complete calculation without
cloning or compiling an external repository. It uses NumPy arrays and a
vectorized bisection solver.

Method
------
The equations and physical constants follow Liljegren et al. (2008) and the
full-radiation implementation described by Kong and Huber (2022). The layout
is independently organized for teaching, with descriptive function names and
unit checks.

Important validation note
-------------------------
Before publication-scale use, benchmark representative cases against a
validated implementation such as PyWBGT and document the comparison.
Small numerical differences can arise from root-solver tolerances and solar
interval conventions.

Inputs use:
- temperature: kelvin
- relative humidity: percent
- pressure: pascals
- wind: m s-1
- radiation: W m-2
- solar cosine and direct fraction: dimensionless

References
----------
Liljegren, J. C., et al. (2008), Journal of Occupational and Environmental
Hygiene, 5, 645-655.
Kong, Q., & Huber, M. (2022), Earth's Future, 10, e2021EF002334.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


# ---------------------------------------------------------------------------
# Physical and sensor constants
# ---------------------------------------------------------------------------

MAIR = 28.97
MH2O = 18.015
RGAS = 8314.34
CP = 1003.5
STEFAN_BOLTZMANN = 5.6696e-8

DIAMETER_GLOBE_M = 0.0508
EMISSIVITY_GLOBE = 0.95
ALBEDO_GLOBE = 0.05

EMISSIVITY_WICK = 0.95
ALBEDO_WICK = 0.40
DIAMETER_WICK_M = 0.007
LENGTH_WICK_M = 0.0254

RATIO = CP * MAIR / MH2O
RAIR = RGAS / MAIR
PRANDTL = CP / (CP + 1.25 * RAIR)

# Stability lookup used to convert 10 m wind to 2 m wind.
_STABILITY_TABLE = np.array(
    [
        [1, 1, 2, 4, 0, 5, 6, 0],
        [1, 2, 3, 4, 0, 5, 6, 0],
        [2, 2, 3, 4, 0, 4, 4, 0],
        [3, 3, 4, 4, 0, 0, 0, 0],
        [3, 4, 4, 4, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ],
    dtype=int,
)
_URBAN_EXPONENT = np.array([0.15, 0.15, 0.20, 0.25, 0.30, 0.30])

# Solar declination constants.
_DECL1 = 0.006918
_DECL2 = 0.399912
_DECL3 = 0.070257
_DECL4 = 0.006758
_DECL5 = 0.000907
_DECL6 = 0.002697
_DECL7 = 0.00148


@dataclass(frozen=True)
class WBGTResult:
    """Container returned by calculate_wbgt."""

    natural_wet_bulb_k: np.ndarray
    globe_temperature_k: np.ndarray
    wbgt_k: np.ndarray
    wind_2m_ms: np.ndarray

    @property
    def natural_wet_bulb_c(self) -> np.ndarray:
        return self.natural_wet_bulb_k - 273.15

    @property
    def globe_temperature_c(self) -> np.ndarray:
        return self.globe_temperature_k - 273.15

    @property
    def wbgt_c(self) -> np.ndarray:
        return self.wbgt_k - 273.15


# ---------------------------------------------------------------------------
# Humidity helpers
# ---------------------------------------------------------------------------

def saturation_vapor_pressure_pa(
    temperature_k,
    pressure_pa,
):
    """Saturation vapor pressure in Pa, including pressure enhancement."""

    temperature_k, pressure_pa = np.broadcast_arrays(
        np.asarray(temperature_k, dtype=float),
        np.asarray(pressure_pa, dtype=float),
    )

    above_freezing = temperature_k > 273.15

    water_value = (
        611.21
        * np.exp(
            17.502
            * (temperature_k - 273.15)
            / (temperature_k - 32.18)
        )
        * (
            1.0007
            + 3.46e-6 * pressure_pa / 100.0
        )
    )

    ice_value = (
        611.15
        * np.exp(
            22.452
            * (temperature_k - 273.15)
            / (temperature_k - 0.6)
        )
        * (
            1.0003
            + 4.18e-6 * pressure_pa / 100.0
        )
    )

    return np.where(
        above_freezing,
        water_value,
        ice_value,
    )


def relative_humidity_from_dewpoint(
    air_temperature_k,
    dewpoint_temperature_k,
    pressure_pa,
):
    """Relative humidity in percent from air and dew-point temperature."""

    e_air = saturation_vapor_pressure_pa(
        air_temperature_k,
        pressure_pa,
    )
    e_dew = saturation_vapor_pressure_pa(
        dewpoint_temperature_k,
        pressure_pa,
    )

    return np.clip(
        100.0 * e_dew / e_air,
        0.0,
        100.0,
    )


def relative_humidity_from_specific_humidity(
    specific_humidity,
    pressure_pa,
    air_temperature_k,
):
    """Relative humidity in percent from specific humidity."""

    q, pressure_pa, air_temperature_k = np.broadcast_arrays(
        np.asarray(specific_humidity, dtype=float),
        np.asarray(pressure_pa, dtype=float),
        np.asarray(air_temperature_k, dtype=float),
    )

    epsilon = 0.622

    vapor_pressure_pa = (
        q * pressure_pa
        / (
            epsilon
            + (1.0 - epsilon) * q
        )
    )

    saturation_pa = saturation_vapor_pressure_pa(
        air_temperature_k,
        pressure_pa,
    )

    return np.clip(
        100.0 * vapor_pressure_pa / saturation_pa,
        0.0,
        100.0,
    )


# ---------------------------------------------------------------------------
# Solar geometry
# ---------------------------------------------------------------------------

def _datetime_parts(times):
    """Return zero-based day of year, fractional UTC hour, and year length."""

    times = np.asarray(times).astype("datetime64[ns]")

    day = times.astype("datetime64[D]")
    year = times.astype("datetime64[Y]")
    year_start = year.astype("datetime64[D]")
    next_year_start = (year + np.timedelta64(1, "Y")).astype("datetime64[D]")

    day_of_year = (
        day - year_start
    ).astype("timedelta64[D]").astype(float)

    seconds_since_midnight = (
        times - day.astype("datetime64[ns]")
    ) / np.timedelta64(1, "s")

    utc_hour = seconds_since_midnight.astype(float) / 3600.0

    year_length = (
        next_year_start - year_start
    ).astype("timedelta64[D]").astype(float)

    return day_of_year, utc_hour, year_length


def solar_declination_rad(times):
    """Solar declination angle in radians."""

    day_of_year, utc_hour, year_length = _datetime_parts(times)

    annual_angle = (
        2.0
        * np.pi
        / year_length
        * (
            day_of_year
            + utc_hour / 24.0
        )
    )

    return (
        _DECL1
        - _DECL2 * np.cos(annual_angle)
        + _DECL3 * np.sin(annual_angle)
        - _DECL4 * np.cos(2.0 * annual_angle)
        + _DECL5 * np.sin(2.0 * annual_angle)
        - _DECL6 * np.cos(3.0 * annual_angle)
        + _DECL7 * np.sin(3.0 * annual_angle)
    )


def _prepare_lat_lon(latitude_deg, longitude_deg):
    """Return 2-D latitude and longitude arrays in radians."""

    latitude_deg = np.asarray(latitude_deg, dtype=float)
    longitude_deg = np.asarray(longitude_deg, dtype=float)

    if latitude_deg.ndim == 1 and longitude_deg.ndim == 1:
        lon_2d, lat_2d = np.meshgrid(
            longitude_deg,
            latitude_deg,
        )
    else:
        lat_2d, lon_2d = np.broadcast_arrays(
            latitude_deg,
            longitude_deg,
        )

    return np.deg2rad(lat_2d), np.deg2rad(lon_2d)


def _integral_cosine(
    start_angle,
    end_angle,
    sin_declination_sin_latitude,
    cos_declination_cos_latitude,
):
    """Integral of A + B cos(hour angle) over an angle interval."""

    return (
        sin_declination_sin_latitude
        * (end_angle - start_angle)
        + cos_declination_cos_latitude
        * (
            np.sin(end_angle)
            - np.sin(start_angle)
        )
    )


def solar_cosines(
    times,
    latitude_deg,
    longitude_deg,
    interval_hours,
):
    """
    Calculate interval-average solar cosines.

    Returns
    -------
    full_interval : ndarray
        Mean cosine zenith angle over the complete interval. It can be
        negative at night.
    sunlit_interval : ndarray
        Mean cosine zenith angle over only the illuminated portion of the
        interval. It is zero when the interval is completely dark.
    """

    times = np.asarray(times).astype("datetime64[ns]")
    lat_rad, lon_rad = _prepare_lat_lon(
        latitude_deg,
        longitude_deg,
    )

    _, utc_hour, _ = _datetime_parts(times)
    declination = solar_declination_rad(times)

    hour_angle_center = (
        (utc_hour[:, None, None] - 12.0)
        * np.pi / 12.0
        + lon_rad[None, :, :]
    )

    half_width = (
        interval_hours
        * np.pi / 24.0
    )

    start = hour_angle_center - half_width
    end = hour_angle_center + half_width

    declination_3d = declination[:, None, None]

    a = (
        np.sin(declination_3d)
        * np.sin(lat_rad)[None, :, :]
    )
    b = (
        np.cos(declination_3d)
        * np.cos(lat_rad)[None, :, :]
    )

    full_integral = _integral_cosine(
        start,
        end,
        a,
        b,
    )
    full_mean = full_integral / (
        2.0 * half_width
    )

    # Daylight exists where A + B cos(h) > 0.
    ratio = np.divide(
        -a,
        b,
        out=np.full_like(a, np.nan),
        where=np.abs(b) > 1e-14,
    )

    polar_day = ratio <= -1.0
    polar_night = ratio >= 1.0
    normal_day = ~(polar_day | polar_night)

    sunrise_magnitude = np.arccos(
        np.clip(
            ratio,
            -1.0,
            1.0,
        )
    )

    daylight_integral = np.zeros_like(full_mean)
    daylight_length = np.zeros_like(full_mean)

    # Cover possible wraparound across +/- pi.
    for period_shift in (
        -2.0 * np.pi,
        0.0,
        2.0 * np.pi,
    ):
        daylight_start = (
            -sunrise_magnitude
            + period_shift
        )
        daylight_end = (
            sunrise_magnitude
            + period_shift
        )

        overlap_start = np.maximum(
            start,
            daylight_start,
        )
        overlap_end = np.minimum(
            end,
            daylight_end,
        )

        overlap_length = np.maximum(
            overlap_end - overlap_start,
            0.0,
        )

        overlap_integral = _integral_cosine(
            overlap_start,
            overlap_end,
            a,
            b,
        )

        daylight_integral += np.where(
            normal_day
            & (overlap_length > 0.0),
            overlap_integral,
            0.0,
        )

        daylight_length += np.where(
            normal_day,
            overlap_length,
            0.0,
        )

    daylight_integral = np.where(
        polar_day,
        full_integral,
        daylight_integral,
    )
    daylight_length = np.where(
        polar_day,
        2.0 * half_width,
        daylight_length,
    )

    daylight_length = np.where(
        polar_night,
        0.0,
        daylight_length,
    )

    sunlit_mean = np.divide(
        daylight_integral,
        daylight_length,
        out=np.zeros_like(daylight_integral),
        where=daylight_length > 0.0,
    )

    sunlit_mean = np.clip(
        sunlit_mean,
        0.0,
        1.0,
    )

    return full_mean, sunlit_mean


# ---------------------------------------------------------------------------
# Air properties, wind-height conversion, and convection
# ---------------------------------------------------------------------------

def air_viscosity_kg_m_s(temperature_k):
    """Dynamic viscosity of air in kg m-1 s-1."""

    temperature_k = np.asarray(
        temperature_k,
        dtype=float,
    )

    omega = (
        1.2945
        - temperature_k / 1141.176470588
    )

    return (
        2.6693e-6
        * np.sqrt(MAIR * temperature_k)
        / (13.082689 * omega)
    )


def air_thermal_conductivity_w_m_k(temperature_k):
    """Thermal conductivity of air in W m-1 K-1."""

    return (
        CP + 1.25 * RAIR
    ) * air_viscosity_kg_m_s(
        temperature_k
    )


def water_vapor_diffusivity_m2_s(
    temperature_k,
    pressure_pa,
):
    """Diffusivity of water vapor in air in m2 s-1."""

    return (
        2.471773765165648e-5
        * (
            np.asarray(
                temperature_k,
                dtype=float,
            )
            * 0.0034210563748421257
        ) ** 2.334
        * (
            np.asarray(
                pressure_pa,
                dtype=float,
            )
            / 101325.0
        ) ** -1.0
    )


def heat_of_evaporation_j_kg(temperature_k):
    """Latent heat term in J kg-1."""

    return (
        1665134.5
        + 2370.0
        * np.asarray(
            temperature_k,
            dtype=float,
        )
    )


def stability_class(
    solar_cosine,
    wind_10m_ms,
    shortwave_down_wm2,
):
    """Atmospheric stability class used for the 10 m to 2 m wind conversion."""

    solar_cosine, wind_10m_ms, shortwave_down_wm2 = np.broadcast_arrays(
        np.asarray(solar_cosine, dtype=float),
        np.asarray(wind_10m_ms, dtype=float),
        np.asarray(shortwave_down_wm2, dtype=float),
    )

    daytime = solar_cosine > 0.0

    radiation_column = np.where(
        shortwave_down_wm2 >= 925.0,
        0,
        np.where(
            shortwave_down_wm2 >= 675.0,
            1,
            np.where(
                shortwave_down_wm2 >= 175.0,
                2,
                3,
            ),
        ),
    )

    daytime_row = np.where(
        wind_10m_ms >= 6.0,
        4,
        np.where(
            wind_10m_ms >= 5.0,
            3,
            np.where(
                wind_10m_ms >= 3.0,
                2,
                np.where(
                    wind_10m_ms >= 2.0,
                    1,
                    0,
                ),
            ),
        ),
    )

    nighttime_row = np.where(
        wind_10m_ms >= 2.5,
        2,
        np.where(
            wind_10m_ms >= 2.0,
            1,
            0,
        ),
    )

    row = np.where(
        daytime,
        daytime_row,
        nighttime_row,
    )
    column = np.where(
        daytime,
        radiation_column,
        5,
    )

    return _STABILITY_TABLE[
        row,
        column,
    ]


def wind_2m_from_10m(
    wind_10m_ms,
    solar_cosine,
    shortwave_down_wm2,
):
    """Convert ERA5 10 m wind speed to the model's 2 m wind speed."""

    wind_10m_ms = np.asarray(
        wind_10m_ms,
        dtype=float,
    )

    class_number = stability_class(
        solar_cosine,
        wind_10m_ms,
        shortwave_down_wm2,
    )

    # The lookup uses classes 1-6. Zero is retained defensively.
    exponent_index = np.clip(
        class_number - 1,
        0,
        len(_URBAN_EXPONENT) - 1,
    )

    exponent = _URBAN_EXPONENT[
        exponent_index
    ]

    return np.maximum(
        wind_10m_ms
        * (2.0 / 10.0) ** exponent,
        0.13,
    )


def sphere_convection_coefficient(
    film_temperature_k,
    pressure_pa,
    wind_2m_ms,
):
    """Convective heat-transfer coefficient for the black globe."""

    thermal_conductivity = (
        air_thermal_conductivity_w_m_k(
            film_temperature_k
        )
    )

    density = (
        pressure_pa
        / (
            RAIR
            * film_temperature_k
        )
    )

    reynolds = (
        wind_2m_ms
        * density
        * DIAMETER_GLOBE_M
        / air_viscosity_kg_m_s(
            film_temperature_k
        )
    )

    nusselt = (
        2.0
        + 0.6
        * np.sqrt(
            np.maximum(
                reynolds,
                0.0,
            )
        )
        * PRANDTL ** 0.3333
    )

    return (
        nusselt
        * thermal_conductivity
        / DIAMETER_GLOBE_M
    )


def cylinder_convection_coefficient(
    film_temperature_k,
    pressure_pa,
    wind_2m_ms,
):
    """Convective heat-transfer coefficient for the wetted wick."""

    thermal_conductivity = (
        air_thermal_conductivity_w_m_k(
            film_temperature_k
        )
    )

    density = (
        pressure_pa
        / (
            RAIR
            * film_temperature_k
        )
    )

    reynolds = (
        wind_2m_ms
        * density
        * DIAMETER_WICK_M
        / air_viscosity_kg_m_s(
            film_temperature_k
        )
    )

    nusselt = (
        0.281
        * np.maximum(
            reynolds,
            0.0,
        ) ** 0.6
        * PRANDTL ** 0.44
    )

    return (
        nusselt
        * thermal_conductivity
        / DIAMETER_WICK_M
    )


# ---------------------------------------------------------------------------
# Numerical solver
# ---------------------------------------------------------------------------

def _bisection_array(
    function: Callable,
    lower,
    upper,
    tolerance_k=0.01,
    max_iterations=100,
):
    """
    Solve many independent scalar roots with a vectorized bisection method.

    Values with missing inputs remain NaN. A clear exception is raised when
    a finite element is not bracketed.
    """

    lower, upper = np.broadcast_arrays(
        np.asarray(lower, dtype=float),
        np.asarray(upper, dtype=float),
    )

    lower = lower.copy()
    upper = upper.copy()

    f_lower = function(lower)
    f_upper = function(upper)

    finite = (
        np.isfinite(lower)
        & np.isfinite(upper)
        & np.isfinite(f_lower)
        & np.isfinite(f_upper)
    )

    not_bracketed = (
        finite
        & (
            np.signbit(f_lower)
            == np.signbit(f_upper)
        )
        & (f_lower != 0.0)
        & (f_upper != 0.0)
    )

    if np.any(not_bracketed):
        count = int(
            np.count_nonzero(
                not_bracketed
            )
        )
        raise ValueError(
            f"{count} finite root(s) were not bracketed. "
            "Check input units and physical ranges."
        )

    solution = np.full(
        lower.shape,
        np.nan,
        dtype=float,
    )

    exact_lower = finite & (f_lower == 0.0)
    exact_upper = finite & (f_upper == 0.0)

    solution[exact_lower] = lower[exact_lower]
    solution[exact_upper] = upper[exact_upper]

    active = (
        finite
        & ~exact_lower
        & ~exact_upper
    )

    for _ in range(max_iterations):
        if not np.any(active):
            break

        midpoint = 0.5 * (
            lower + upper
        )
        f_midpoint = function(midpoint)

        left_half = (
            np.signbit(f_lower)
            != np.signbit(f_midpoint)
        )

        upper = np.where(
            active & left_half,
            midpoint,
            upper,
        )
        f_upper = np.where(
            active & left_half,
            f_midpoint,
            f_upper,
        )

        lower = np.where(
            active & ~left_half,
            midpoint,
            lower,
        )
        f_lower = np.where(
            active & ~left_half,
            f_midpoint,
            f_lower,
        )

        converged = (
            active
            & (
                np.abs(
                    upper - lower
                )
                <= tolerance_k
            )
        )

        solution[converged] = (
            0.5
            * (
                lower[converged]
                + upper[converged]
            )
        )

        active = active & ~converged

    if np.any(active):
        solution[active] = (
            0.5
            * (
                lower[active]
                + upper[active]
            )
        )

    return solution


# ---------------------------------------------------------------------------
# Explicit globe and natural wet-bulb temperatures
# ---------------------------------------------------------------------------

def globe_temperature_k(
    air_temperature_k,
    pressure_pa,
    wind_ms,
    shortwave_down_wm2,
    shortwave_up_wm2,
    longwave_down_wm2,
    longwave_up_wm2,
    direct_fraction,
    sunlit_cosine,
    wind_height_m=10,
    tolerance_k=0.01,
):
    """Explicit black-globe temperature in kelvin."""

    arrays = np.broadcast_arrays(
        np.asarray(air_temperature_k, dtype=float),
        np.asarray(pressure_pa, dtype=float),
        np.asarray(wind_ms, dtype=float),
        np.asarray(shortwave_down_wm2, dtype=float),
        np.asarray(shortwave_up_wm2, dtype=float),
        np.asarray(longwave_down_wm2, dtype=float),
        np.asarray(longwave_up_wm2, dtype=float),
        np.asarray(direct_fraction, dtype=float),
        np.asarray(sunlit_cosine, dtype=float),
    )

    (
        air_temperature_k,
        pressure_pa,
        wind_ms,
        shortwave_down_wm2,
        shortwave_up_wm2,
        longwave_down_wm2,
        longwave_up_wm2,
        direct_fraction,
        sunlit_cosine,
    ) = arrays

    direct_fraction = np.clip(
        direct_fraction,
        0.0,
        0.9,
    )

    safe_cosine = np.where(
        sunlit_cosine > 0.0,
        sunlit_cosine,
        1.0,
    )

    direct_fraction = np.where(
        sunlit_cosine > 0.0,
        direct_fraction,
        0.0,
    )

    if wind_height_m == 10:
        wind_2m_ms = wind_2m_from_10m(
            wind_ms,
            sunlit_cosine,
            shortwave_down_wm2,
        )
    elif wind_height_m == 2:
        wind_2m_ms = np.maximum(
            wind_ms,
            0.13,
        )
    else:
        raise ValueError(
            "wind_height_m must be 2 or 10."
        )

    radiative_constant = (
        0.5
        * (
            longwave_down_wm2
            + longwave_up_wm2
        )
        / STEFAN_BOLTZMANN
        + shortwave_down_wm2
        * (
            1.0
            - ALBEDO_GLOBE
        )
        / (
            2.0
            * EMISSIVITY_GLOBE
            * STEFAN_BOLTZMANN
        )
        * (
            1.0
            - direct_fraction
            + 0.5
            * direct_fraction
            / safe_cosine
        )
        + (
            1.0
            - ALBEDO_GLOBE
        )
        * shortwave_up_wm2
        / (
            2.0
            * EMISSIVITY_GLOBE
            * STEFAN_BOLTZMANN
        )
    )

    def residual(globe_temperature):
        film_temperature = (
            0.5
            * (
                air_temperature_k
                + globe_temperature
            )
        )

        convection = sphere_convection_coefficient(
            film_temperature,
            pressure_pa,
            wind_2m_ms,
        )

        return (
            radiative_constant
            - convection
            * (
                globe_temperature
                - air_temperature_k
            )
            / (
                EMISSIVITY_GLOBE
                * STEFAN_BOLTZMANN
            )
            - globe_temperature ** 4
        )

    lower = air_temperature_k - 50.0
    upper = air_temperature_k + 90.0

    return _bisection_array(
        residual,
        lower,
        upper,
        tolerance_k=tolerance_k,
    )


def natural_wet_bulb_temperature_k(
    air_temperature_k,
    relative_humidity_pct,
    pressure_pa,
    wind_ms,
    shortwave_down_wm2,
    shortwave_up_wm2,
    longwave_down_wm2,
    longwave_up_wm2,
    direct_fraction,
    sunlit_cosine,
    wind_height_m=10,
    tolerance_k=0.01,
):
    """Explicit natural wet-bulb temperature in kelvin."""

    arrays = np.broadcast_arrays(
        np.asarray(air_temperature_k, dtype=float),
        np.asarray(relative_humidity_pct, dtype=float),
        np.asarray(pressure_pa, dtype=float),
        np.asarray(wind_ms, dtype=float),
        np.asarray(shortwave_down_wm2, dtype=float),
        np.asarray(shortwave_up_wm2, dtype=float),
        np.asarray(longwave_down_wm2, dtype=float),
        np.asarray(longwave_up_wm2, dtype=float),
        np.asarray(direct_fraction, dtype=float),
        np.asarray(sunlit_cosine, dtype=float),
    )

    (
        air_temperature_k,
        relative_humidity_pct,
        pressure_pa,
        wind_ms,
        shortwave_down_wm2,
        shortwave_up_wm2,
        longwave_down_wm2,
        longwave_up_wm2,
        direct_fraction,
        sunlit_cosine,
    ) = arrays

    relative_humidity_pct = np.clip(
        relative_humidity_pct,
        0.0,
        100.0,
    )

    direct_fraction = np.clip(
        direct_fraction,
        0.0,
        0.9,
    )

    safe_cosine = np.where(
        sunlit_cosine > 0.0,
        sunlit_cosine,
        1.0,
    )

    direct_fraction = np.where(
        sunlit_cosine > 0.0,
        direct_fraction,
        0.0,
    )

    if wind_height_m == 10:
        wind_2m_ms = wind_2m_from_10m(
            wind_ms,
            sunlit_cosine,
            shortwave_down_wm2,
        )
    elif wind_height_m == 2:
        wind_2m_ms = np.maximum(
            wind_ms,
            0.13,
        )
    else:
        raise ValueError(
            "wind_height_m must be 2 or 10."
        )

    ambient_vapor_pressure_pa = (
        relative_humidity_pct
        / 100.0
        * saturation_vapor_pressure_pa(
            air_temperature_k,
            pressure_pa,
        )
    )

    wick_geometry = (
        DIAMETER_WICK_M
        / (
            4.0
            * LENGTH_WICK_M
        )
    )

    zenith_angle = np.arccos(
        np.clip(
            safe_cosine,
            0.0,
            1.0,
        )
    )

    radiative_gain = (
        EMISSIVITY_WICK
        * 0.5
        * (
            longwave_down_wm2
            + longwave_up_wm2
        )
        + (
            1.0
            + wick_geometry
        )
        * (
            1.0
            - ALBEDO_WICK
        )
        * (
            1.0
            - direct_fraction
        )
        * shortwave_down_wm2
        + (
            np.tan(
                zenith_angle
            )
            / np.pi
            + wick_geometry
        )
        * (
            1.0
            - ALBEDO_WICK
        )
        * direct_fraction
        * shortwave_down_wm2
        + (
            1.0
            - ALBEDO_WICK
        )
        * shortwave_up_wm2
    )

    def residual(wet_bulb_temperature):
        film_temperature = (
            0.5
            * (
                air_temperature_k
                + wet_bulb_temperature
            )
        )

        evaporation_heat = heat_of_evaporation_j_kg(
            film_temperature
        )

        saturation_at_wick = saturation_vapor_pressure_pa(
            wet_bulb_temperature,
            pressure_pa,
        )

        density = (
            pressure_pa
            / (
                RAIR
                * film_temperature
            )
        )

        schmidt = (
            air_viscosity_kg_m_s(
                film_temperature
            )
            / (
                density
                * water_vapor_diffusivity_m2_s(
                    film_temperature,
                    pressure_pa,
                )
            )
        )

        convection = cylinder_convection_coefficient(
            film_temperature,
            pressure_pa,
            wind_2m_ms,
        )

        net_radiation = (
            radiative_gain
            - EMISSIVITY_WICK
            * STEFAN_BOLTZMANN
            * wet_bulb_temperature ** 4
        )

        return (
            air_temperature_k
            - evaporation_heat
            / RATIO
            * (
                saturation_at_wick
                - ambient_vapor_pressure_pa
            )
            / (
                pressure_pa
                - saturation_at_wick
            )
            * (
                PRANDTL
                / schmidt
            ) ** 0.56
            + net_radiation
            / convection
            - wet_bulb_temperature
        )

    lower = (
        air_temperature_k
        - (
            100.0
            - relative_humidity_pct
        )
        / 5.0
        - 50.0
    )

    upper = np.minimum(
        air_temperature_k + 70.0,
        340.0,
    )

    return _bisection_array(
        residual,
        lower,
        upper,
        tolerance_k=tolerance_k,
    )


def calculate_wbgt(
    air_temperature_k,
    relative_humidity_pct,
    pressure_pa,
    wind_ms,
    shortwave_down_wm2,
    shortwave_up_wm2,
    longwave_down_wm2,
    longwave_up_wm2,
    direct_fraction,
    sunlit_cosine,
    wind_height_m=10,
    tolerance_k=0.01,
):
    """Calculate natural wet-bulb, globe, and outdoor WBGT temperatures."""

    globe_k = globe_temperature_k(
        air_temperature_k=air_temperature_k,
        pressure_pa=pressure_pa,
        wind_ms=wind_ms,
        shortwave_down_wm2=shortwave_down_wm2,
        shortwave_up_wm2=shortwave_up_wm2,
        longwave_down_wm2=longwave_down_wm2,
        longwave_up_wm2=longwave_up_wm2,
        direct_fraction=direct_fraction,
        sunlit_cosine=sunlit_cosine,
        wind_height_m=wind_height_m,
        tolerance_k=tolerance_k,
    )

    natural_wet_bulb_k = natural_wet_bulb_temperature_k(
        air_temperature_k=air_temperature_k,
        relative_humidity_pct=relative_humidity_pct,
        pressure_pa=pressure_pa,
        wind_ms=wind_ms,
        shortwave_down_wm2=shortwave_down_wm2,
        shortwave_up_wm2=shortwave_up_wm2,
        longwave_down_wm2=longwave_down_wm2,
        longwave_up_wm2=longwave_up_wm2,
        direct_fraction=direct_fraction,
        sunlit_cosine=sunlit_cosine,
        wind_height_m=wind_height_m,
        tolerance_k=tolerance_k,
    )

    wbgt_k = (
        0.7 * natural_wet_bulb_k
        + 0.2 * globe_k
        + 0.1
        * np.asarray(
            air_temperature_k,
            dtype=float,
        )
    )

    if wind_height_m == 10:
        wind_2m_ms = wind_2m_from_10m(
            wind_ms,
            sunlit_cosine,
            shortwave_down_wm2,
        )
    else:
        wind_2m_ms = np.maximum(
            np.asarray(
                wind_ms,
                dtype=float,
            ),
            0.13,
        )

    return WBGTResult(
        natural_wet_bulb_k=natural_wet_bulb_k,
        globe_temperature_k=globe_k,
        wbgt_k=wbgt_k,
        wind_2m_ms=wind_2m_ms,
    )


# ---------------------------------------------------------------------------
# Approximate indices and labor response
# ---------------------------------------------------------------------------

def simplified_wbgt_c(
    air_temperature_c,
    relative_humidity_pct,
):
    """Simplified WBGT approximation in degrees C."""

    air_temperature_k = (
        np.asarray(
            air_temperature_c,
            dtype=float,
        )
        + 273.15
    )
    pressure_pa = np.full_like(
        air_temperature_k,
        101325.0,
    )

    vapor_pressure_hpa = (
        np.asarray(
            relative_humidity_pct,
            dtype=float,
        )
        / 100.0
        * saturation_vapor_pressure_pa(
            air_temperature_k,
            pressure_pa,
        )
        / 100.0
    )

    return (
        0.567
        * np.asarray(
            air_temperature_c,
            dtype=float,
        )
        + 0.393
        * vapor_pressure_hpa
        + 3.94
    )


def environmental_stress_index_c(
    air_temperature_c,
    relative_humidity_pct,
    shortwave_down_wm2,
):
    """Environmental Stress Index approximation in degrees C."""

    air_temperature_c = np.asarray(
        air_temperature_c,
        dtype=float,
    )
    relative_humidity_fraction = (
        np.asarray(
            relative_humidity_pct,
            dtype=float,
        )
        / 100.0
    )
    shortwave_down_wm2 = np.asarray(
        shortwave_down_wm2,
        dtype=float,
    )

    return (
        0.62 * air_temperature_c
        - 0.7
        * relative_humidity_fraction
        + 0.002
        * shortwave_down_wm2
        + 0.43
        * air_temperature_c
        * relative_humidity_fraction
        - 0.078
        / (
            0.1
            + shortwave_down_wm2
        )
    )


def iso_wbgt_limit_c(
    metabolic_rate_w,
):
    """ISO-style WBGT limit in degrees C."""

    return (
        56.7
        - 11.5
        * np.log10(
            metabolic_rate_w
        )
    )


def iso_labor_productivity(
    wbgt_c,
    metabolic_rate_w,
):
    """Work fraction from zero to one."""

    work_limit_c = iso_wbgt_limit_c(
        metabolic_rate_w
    )
    rest_limit_c = iso_wbgt_limit_c(
        117.0
    )

    work_fraction = (
        rest_limit_c
        - np.asarray(
            wbgt_c,
            dtype=float,
        )
    ) / (
        rest_limit_c
        - work_limit_c
    )

    return np.clip(
        work_fraction,
        0.0,
        1.0,
    )
