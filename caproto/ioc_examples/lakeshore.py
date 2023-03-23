"""
An IOC with a simulated temperature controller.
This demonstrates how to do put_completion with a caproto ioc and ophyd device.

Another interesting part of this example is the use of settle_time.
Settle_time allows us to wait an additional amount after the put_completion.
It is used here to wait for the stabilization of the material temperature
after the setpoint ramp is complete.

This also includes a simulated material that can be heated/cooled and a
PID controller that can be connected to arbitrary systems. The PIDController
has a ramp feature so that the setpoint ramps to the target value gradually.
"""

import asyncio
import contextvars
import logging
import functools
import time

from simple_pid import PID
from caproto.server import PVGroup, ioc_arg_parser, pvproperty, run
from ophyd import EpicsSignal, EpicsSignalRO, PVPositionerPC
from ophyd import Component as Cpt


logger = logging.getLogger(__name__)


class ThermalMaterial:
    """
    A material that you can heat and cool.

    Parameters
    ----------
    thermal_mass: float, optional
        How the material's energy relates to its temperature.
    start_temp: float, optional
        Starting temperature of the material.
    ambient_temp: float, optional
        The temperature of the environment.
    heater_power: float, optional
        The rate at which the heater is adding energy to the sample.
    cooling_constant: float, optional
        How readily the sample releases energy to the environment.
    """

    def __init__(
        self,
        thermal_mass=100,
        start_temp=100,
        ambient_temp=0,
        heater_power=0,
        cooling_constant=1,
    ):
        self.energy = start_temp * thermal_mass
        self.thermal_mass = thermal_mass
        self.ambient_temp = ambient_temp
        self._heater_power = heater_power
        self.cooling_constant = cooling_constant
        self.time = time.time()
        self._run = True

    @property
    def temperature(self):
        return self.energy / self.thermal_mass

    @property
    def heater_power(self):
        return self._heater_power

    @heater_power.setter
    def heater_power(self, value):
        self._heater_power = value

    def set_heater_power(self, value):
        self._heater_power = value

    def stop(self):
        self._run = False

    def _cooling(self):
        return -1 * self.cooling_constant * (self.temperature - self.ambient_temp)

    def _heating(self):
        return self.heater_power

    def simulate(self, dt: float):
        self.energy += self._cooling() * dt
        self.energy += self._heating() * dt


class PIDController(PID):
    """
    General purpose PID controller that supports ramping.

    Parameters
    ----------
    get_feedback: callable
        A function that returns the feedback value of the system.
    set_output: callable
        A function that sets the output value of the system.
    ramp_rate: int, float
        The rate that the setpoint should ramp at.
    setpoint: int, float

    Example
    -------
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from caproto.ioc_examples.lakeshore import (PIDController,
    ThermalMaterial)

    sample = ThermalMaterial()
    setpoint = 150

    temperature_controller = PIDController(
        lambda: sample.temperature,
        sample.set_heater_power,
        Kp=1,
        Ki=0.1,
        Kd=0.05,
        ramp_rate=1,
        setpoint=setpoint,
    )

    def update(i):
        x_values = temperature_controller.history['timestamp']
        plt.cla()
        plt.plot(x_values, temperature_controller.history['feedback'])
        plt.plot(x_values, temperature_controller.history['output'])
        plt.plot(x_values, temperature_controller.history['setpoint'])
        plt.xlabel('time')
        plt.ylabel('temperature')
        plt.title('Temperature Controller')
        plt.gcf().autofmt_xdate()
        plt.tight_layout()

    ani = FuncAnimation(plt.gcf(), update, 1000)
    plt.tight_layout()
    plt.show(block=False)
    """

    def __init__(
        self, get_feedback, set_output, ramp_rate=1, setpoint=150, *args, **kwargs
    ):
        self._setpoint = setpoint
        self.ramp_rate = ramp_rate
        super().__init__(setpoint=setpoint, *args, **kwargs)
        self._get_feedback = get_feedback
        self._set_output = set_output
        self._feedback = None
        self._output = None
        self._run = True
        self._ramping = False
        self.history = {"output": [], "feedback": [], "setpoint": [], "timestamp": []}

    @property
    def output(self):
        return self._output

    @property
    def feedback(self):
        return self._feedback

    @property
    def setpoint(self):
        return self._setpoint

    @property
    def setpoint_target(self):
        return self._setpoint_target

    @setpoint.setter
    def setpoint(self, value):
        """
        Always set ramping to true at the same time as
        the setpoint change. Is will avoid a race condition,
        and will allow the client to just check for ramping = False
        to determine completion.
        """
        self._setpoint_target = value
        self._ramping = True
        if self.ramp_rate is None:
            self._setpoint = value

    @property
    def ramping(self):
        return self._ramping

    def stop(self):
        self._run = False

    def simulate(self, dt: float):
        self._feedback = self._get_feedback()
        self._output = self.__call__(self._get_feedback())
        self.history["output"].append(self._output)
        self.history["feedback"].append(self._feedback)
        self.history["setpoint"].append(self.setpoint)
        self.history["timestamp"].append(time.time())
        self._set_output(self._output)

        # Ramping logic.
        remaining = self._setpoint_target - self.setpoint
        self._ramping = bool(remaining)
        if isinstance(self.ramp_rate, (int, float)):
            if remaining > 0:
                self._setpoint += min(self.ramp_rate * dt, abs(remaining))
            elif remaining < 0:
                self._setpoint -= min(self.ramp_rate * dt, abs(remaining))
        elif self.ramp_rate is None and remaining != 0:
            self._setpoint = self._setpoint_target


internal_process = contextvars.ContextVar("internal_process", default=False)


def no_reentry(func):
    """
    This is needed for put completion.
    """

    @functools.wraps(func)
    async def inner(*args, **kwargs):
        if internal_process.get():
            return
        try:
            internal_process.set(True)
            return await func(*args, **kwargs)
        finally:
            internal_process.set(False)

    return inner


class LakeshoreIOC(PVGroup):
    """
    Simulated Lakeshore IOC with put completion on the setpoint.
    """

    Kp = pvproperty(value=1.0, doc="PID parameter Kp")
    Ki = pvproperty(value=0.1, doc="PID parameter Ki")
    Kd = pvproperty(value=0, doc="PID parameter Kd")
    ramp_rate = pvproperty(value=1.0, doc="Ramp rate")
    setpoint = pvproperty(value=100.0, doc="temperature setpoint")
    feedback = pvproperty(value=100.0, doc="temperature feedback")
    output = pvproperty(value=100.0, doc="output value")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._sample = ThermalMaterial()

        self._temperature_controller = PIDController(
            lambda: self._sample.temperature,
            self._sample.set_heater_power,
            Kp=1,
            Ki=0.1,
            Kd=0.05,
            setpoint=150,
        )
        self._last_tick = time.monotonic()

    async def __ainit__(self, async_lib):
        self._pc_event = async_lib.Event()

    async def simulate(self, dt: float):
        # Update internal parameters
        self._temperature_controller.Kd = self.Kd.value
        self._temperature_controller.Ki = self.Ki.value
        self._temperature_controller.Kp = self.Kp.value
        self._temperature_controller.ramp_rate = self.ramp_rate.value
        self._temperature_controller.setpoint = self.setpoint.value
        # Simulate
        self._sample.simulate(dt)
        self._temperature_controller.simulate(dt)
        # Update outputs
        await self.feedback.write(self._temperature_controller.feedback)
        await self.output.write(self._temperature_controller.output)
        # Debug
        logger.warning(
            f"Simulate step dt={dt:.1f}s "
            f"ramp_rate={self._temperature_controller.ramp_rate:.3f} "
            f"Kp={self._temperature_controller.Kp:.3f} "
            f"Ki={self._temperature_controller.Ki:.3f} "
            f"Kd={self._temperature_controller.Kd:.3f} "
            f"setpoint_target={self._temperature_controller.setpoint_target:.3f} "
            f"setpoint={self._temperature_controller.setpoint:.3f} "
            f"feedback={self.feedback.value:.3f} "
            f"output={self.output.value:.3f}"
        )

    async def wait_for_completion(self):
        """
        Wait until the device is done changing the setpoint.
        """
        while True:
            if not self._temperature_controller.ramping:
                return
            await asyncio.sleep(0.1)

    @setpoint.putter
    @no_reentry
    async def setpoint(self, instance, value):
        logger.warning("New setpoint: %s", value)
        await self.setpoint.write(value)

        ev = self._pc_event
        if not ev.is_set():
            await ev.wait()
            return self.setpoint.value

        ev.clear()
        try:
            await self.wait_for_completion()
        finally:
            ev.set()
        return self.setpoint.value

    @setpoint.scan(period=0.1)
    async def setpoint(self, instance, async_lib):
        """
        This is needed to enable put completion.
        """
        dt = time.monotonic() - self._last_tick
        self._last_tick = time.monotonic()
        await self.simulate(dt)


class Lakeshore(PVPositionerPC):
    """
    Example Ophyd device for Lakeshore that uses put completion.
    PVPositionerPC does not require a done signal like PVPositioner,
    instead it uses the setpoint put_completion.

    Example
    -------
    ls = Lakeshore('Lakeshore', name='Lakeshore', settle_time=5)
    ls.set(100).wait()

    This will wait for the ramp to be completed and also wait for
    the settle_time.
    """

    feedback = Cpt(EpicsSignalRO, ":feedback")
    output = Cpt(EpicsSignalRO, ":output")
    setpoint = Cpt(EpicsSignal, ":setpoint", put_complete=True)
    ramp_rate = Cpt(EpicsSignal, ":ramp_rate")


if __name__ == "__main__":
    ioc_options, run_options = ioc_arg_parser(
        default_prefix="Lakeshore:", desc="Lakeshore IOC"
    )
    ioc = LakeshoreIOC(**ioc_options)

    print("PVs:", list(ioc.pvdb))
    run(ioc.pvdb, startup_hook=ioc.__ainit__, **run_options)
