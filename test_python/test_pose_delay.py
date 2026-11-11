#!/usr/bin/env python
"""The delay-compensated external pose update, on the firmware's own Kalman
core built for the host with plain gcc (no SWIG bindings needed):

    python -m pytest test_python/test_pose_delay.py

test_python/host/pose_delay_host.c flies a straight line at constant speed,
yawing at a constant rate, with mocap poses that arrive a fixed latency
late, through the same loop estimator_kalman.c runs.
"""
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
INCLUDE = [
    "src/modules/interface", "src/modules/interface/kalman_core", "src/hal/interface",
    "src/utils/interface", "src/utils/interface/lighthouse", "src/config",
    "src/drivers/interface", "src/platform/interface",
    "vendor/CMSIS/CMSIS/DSP/Include", "vendor/CMSIS/CMSIS/Core/Include",
]
SOURCES = [
    "test_python/host/pose_delay_host.c",
    "src/modules/src/kalman_core/kalman_core.c",
    "src/modules/src/kalman_core/mm_pose.c",
    "src/modules/src/kalman_core/pose_history.c",
    "vendor/CMSIS/CMSIS/DSP/Source/MatrixFunctions/arm_mat_mult_f32.c",
    "vendor/CMSIS/CMSIS/DSP/Source/MatrixFunctions/arm_mat_trans_f32.c",
    "vendor/CMSIS/CMSIS/DSP/Source/MatrixFunctions/arm_mat_scale_f32.c",
    "vendor/CMSIS/CMSIS/DSP/Source/BasicMathFunctions/arm_scale_f32.c",
    "vendor/CMSIS/CMSIS/DSP/Source/CommonTables/arm_common_tables.c",
    "vendor/CMSIS/CMSIS/DSP/Source/FastMathFunctions/arm_cos_f32.c",
    "vendor/CMSIS/CMSIS/DSP/Source/FastMathFunctions/arm_sin_f32.c",
]

SPEED = 10.0          # m/s
LATENCY_US = 15000    # 15 ms: 15 cm of lag at 10 m/s


@pytest.fixture(scope="module")
def host(tmp_path_factory):
    cc = shutil.which("gcc") or shutil.which("cc")
    if cc is None:
        pytest.skip("no host C compiler")
    tmp = tmp_path_factory.mktemp("pose_delay")
    (tmp / "autoconf.h").write_text("/* host build: no Kconfig */\n")
    exe = tmp / "pose_delay_host"
    subprocess.run(
        [cc, "-O2", "-w", "-DUNIT_TEST_MODE", "-fno-strict-aliasing", "-I", str(tmp),
         *[f"-I{ROOT / d}" for d in INCLUDE], *[str(ROOT / s) for s in SOURCES],
         "-lm", "-o", str(exe)], check=True)
    return exe


def run_flight(host, mode, speed=SPEED, yaw_rate=0.0, latency_us=LATENCY_US, age_us=LATENCY_US,
               offset=0.0, dump=False):
    args = [str(host), "flight", mode, str(speed), str(yaw_rate), str(latency_us), str(age_us),
            str(offset)]
    out = subprocess.run(args + (["dump"] if dump else []), capture_output=True, text=True,
                         check=True).stdout.splitlines()
    pos_err, yaw_err, drops, worst = out[0].split()
    return float(pos_err), float(yaw_err), int(drops), (out[1] if dump else None), float(worst)


def flight(*args, **kwargs):
    return run_flight(*args, **kwargs)[:4]


def test_history_ring(host):
    out = subprocess.run([str(host), "ring"], capture_output=True, text=True, check=True).stdout
    checks = dict(line.split() for line in out.splitlines())
    assert len(checks) >= 10
    failed = [name for name, ok in checks.items() if ok != "1"]
    assert not failed, failed


def test_stock_update_lags_by_speed_times_latency(host):
    pos_err, _, _, _ = flight(host, "stock")
    assert pos_err == pytest.approx(-SPEED * LATENCY_US * 1e-6, abs=0.01)


def test_compensated_update_removes_the_lag(host):
    pos_err, _, drops, _ = flight(host, "comp")
    assert abs(pos_err) < 0.005
    assert drops == 0


def test_age_zero_is_the_stock_update_bit_for_bit(host):
    *_, stock = flight(host, "stock", dump=True)
    *_, comp = flight(host, "comp", age_us=0, dump=True)
    assert stock == comp


@pytest.mark.parametrize("latency_us", [5000, 25000, 40000, 60000])
def test_stable_when_latency_spans_several_poses(host, latency_us):
    # Poses arrive every 8.3 ms. Past that latency each pose is compared with
    # a state that predates the previous correction; the history shift is
    # what keeps that from counting a correction more than once.
    pos_err, _, drops, _ = flight(host, "comp", latency_us=latency_us, age_us=latency_us)
    assert abs(pos_err) < 0.005
    # only the first poses after start-up, before the history is that long
    assert drops <= latency_us // 8333 + 1


def test_wrong_age_costs_speed_times_the_error(host):
    # told 10 ms, really 15: the remaining 5 ms shows up as lag
    pos_err, _, _, _ = flight(host, "comp", age_us=10000)
    assert pos_err == pytest.approx(-SPEED * 0.005, abs=0.01)
    # told 20 ms: the estimate leads
    pos_err, _, _, _ = flight(host, "comp", age_us=20000)
    assert pos_err == pytest.approx(SPEED * 0.005, abs=0.01)


def test_fast_yaw_heading_lag_removed(host):
    yaw_rate = 6.0   # rad/s: 90 mrad of heading lag at 15 ms
    _, stock_yaw, _, _ = flight(host, "stock", yaw_rate=yaw_rate)
    _, comp_yaw, _, _ = flight(host, "comp", yaw_rate=yaw_rate)
    assert stock_yaw == pytest.approx(-yaw_rate * LATENCY_US * 1e-6, abs=0.02)
    assert abs(comp_yaw) < 0.01
    # and the position lag is removed through the turn as well: what is left
    # is the filter's own error while yawing, which poses with no latency at
    # all show too
    stock_pos, _, _, _ = flight(host, "stock", yaw_rate=yaw_rate)
    comp_pos, _, _, _ = flight(host, "comp", yaw_rate=yaw_rate)
    undelayed_pos, _, _, _ = flight(host, "stock", yaw_rate=yaw_rate, latency_us=0)
    assert stock_pos - undelayed_pos == pytest.approx(-SPEED * LATENCY_US * 1e-6, abs=0.01)
    assert abs(comp_pos - undelayed_pos) < 0.01


def test_older_than_history_is_dropped(host):
    _, _, drops, _ = flight(host, "comp", latency_us=80000, age_us=80000)
    assert drops > 100


@pytest.mark.parametrize("latency_us", [15000, 40000, 60000])
def test_converges_without_overshoot(host, latency_us):
    # The estimate starts 30 cm off. Each pose is older than the interval
    # between poses, so without carrying every correction into the history
    # the same error is corrected again and again and the estimate overshoots
    # by 14 to 50 cm here; with it there is none to speak of.
    pos_err, _, _, _, worst = run_flight(host, "comp", latency_us=latency_us, age_us=latency_us,
                                         offset=0.3)
    assert abs(pos_err) < 0.005
    assert worst < 0.01
