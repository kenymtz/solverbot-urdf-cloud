"""tools/wsl/sim_env.sh con un venv real y un ROS falso: cargar el entorno no debe perder `ros2` del PATH."""
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SIM_ENV = Path(__file__).resolve().parents[2] / 'tools' / 'wsl' / 'sim_env.sh'
ACTIVATE = 'source "$HOME/solverbot_venv/bin/activate"'
pytestmark = pytest.mark.skipif(shutil.which('bash') is None, reason='requiere bash')


@pytest.fixture
def home(tmp_path):
    subprocess.run([sys.executable, '-m', 'venv', '--without-pip', str(tmp_path / 'solverbot_venv')], check=True)
    ros = tmp_path / 'ros'
    (ros / 'bin').mkdir(parents=True)
    (ros / 'bin' / 'ros2').write_text('#!/bin/sh\n')
    (ros / 'bin' / 'ros2').chmod(0o755)
    (ros / 'setup.bash').write_text(f'export PATH="{ros}/bin:$PATH"\n')
    return tmp_path


def load(home, *steps):
    """Ejecuta los pasos en una shell limpia y devuelve (ruta de ros2, VIRTUAL_ENV)."""
    script = '; '.join([*steps, 'echo "ros2=$(command -v ros2)"', 'echo "venv=$VIRTUAL_ENV"'])
    res = subprocess.run(['bash', '--noprofile', '--norc', '-c', script], capture_output=True, text=True, check=True,
                         env={'HOME': str(home), 'PATH': '/usr/bin:/bin', 'SOLVERBOT_ROS_SETUP': str(home / 'ros' / 'setup.bash')})
    out = dict(line.split('=', 1) for line in res.stdout.splitlines() if line.startswith(('ros2=', 'venv=')))
    return out['ros2'], out['venv']


def test_terminal_nueva(home):
    assert load(home, f'source "{SIM_ENV}" 0') == (str(home / 'ros' / 'bin' / 'ros2'), str(home / 'solverbot_venv'))


def test_venv_ya_activo_no_pierde_ros2(home):
    assert load(home, ACTIVATE, f'source "{SIM_ENV}" 0') == (str(home / 'ros' / 'bin' / 'ros2'), str(home / 'solverbot_venv'))


def test_cargar_dos_veces_no_pierde_ros2(home):
    assert load(home, f'source "{SIM_ENV}" 0', f'source "{SIM_ENV}" 0')[0] == str(home / 'ros' / 'bin' / 'ros2')
