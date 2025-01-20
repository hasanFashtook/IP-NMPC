import casadi.casadi as cs
import opengen as og
import matplotlib.pyplot as plt
import numpy as np
import math

DZcv = 0.00793711
Fc =2.28133
Fs =2.53165
Xc =0.9
Yc =2.3

m1=1.12
m2=0.12
l=0.033245968
J = 0.013935418
fp = 0.000107443
a = l**2 + J/(m1 + m2)
mi = (m1 + m2)*l
g=9.81

sampling_time = 0.01
nx=4
N=25



def dynamics_ct(x, u):
    Tc = cs.if_else(x[2] < -Xc, 
                    ((Yc - Fc) / Xc) * x[2] - Fc,
                    cs.if_else(cs.logic_and(x[2] >= -Xc, x[2] < -DZcv),
                               ((Yc - Fs) / (Xc**2)) * (x[2]**2) + 2 * ((Yc - Fs) / Xc) * x[2] - Fs,
                               cs.if_else(cs.logic_and(x[2] >= -DZcv, x[2] <= DZcv),
                                          (Fs / DZcv) * x[2],
                                          cs.if_else(cs.logic_and(x[2] > DZcv, x[2] < Xc),
                                                     ((Fs - Yc) / (Xc**2)) * (x[2]**2) - 2 * ((Fs - Yc) / Xc) * x[2] + Fs,
                                                     cs.if_else(x[2] >= Xc,
                                                                ((Yc - Fc) / Xc) * x[2] + Fc, 
                                                                0)))))
    t1 = cs.cos(x[3])
    t2 = cs.sin(x[3])
    t3 = t2**2
    t4 = J + mi * l * t3
    t5 = 1.0 / t4
    t6 = x[2]**2
    t7 = Tc
    t8 = mi * t2
    t9 = t8 * t6
    t10 = u - t7 - t9
    t11 = l * t1
    t12 = fp * x[2]
    
    dx1 = x[1]
    dx2 = t5 * (a * t10 + t11 * (g * t8 - t12))
    dx3 = x[3]
    dx4 = t5 * (t11 * t10 + g * t8 - t12)
    return [dx1, dx2, dx3, dx4]

def dynamics_dt(x, u):
    dx = dynamics_ct(x, u)
    return [x[i] + sampling_time * dx[i] for i in range(nx)]


def stage_cost(x, u):
    # تعديلات على الكلفة المرحلية لنظام inverted pendulum
    cost = 50*x[0]**2 + 1*x[1]**2 + 500*x[2]**2 + 1*x[3]**2 + 0.1*u**2
    return cost

def terminal_cost(x):
    # تعديلات على الكلفة النهائية لنظام inverted pendulum
    cost = 500*x[0]**2 + 50*x[1]**2 + 2000*x[2]**2 + 50*x[3]**2
    return cost

# state_constraints = og.constraints.Rectangle( 
#     xmin=[-np.inf, -1.0, -np.inf, -np.pi],  # الحدود الدنيا لمتحولات الحالة
#     xmax=[np.inf, 1.0, np.inf, np.pi]       # الحدود العليا لمتحولات الحالة
# )

# # تعريف حدود شعاع التحكم
# control_constraints = og.constraints.Rectangle(
#     xmin=[-9.4],  # الحد الأدنى لشعاع التحكم
#     xmax=[9.4]    # الحد الأقصى لشعاع التحكم
# )

# تعريف حدود متحولات الحالة
state_constraints = og.constraints.Rectangle(
    xmin=[-1.0, -np.inf, -np.pi, -np.inf],  # السماح بحركة النواس بالكامل
    xmax=[1.0, np.inf, np.pi, np.inf]       # السماح بحركة النواس بالكامل
)

# تعريف حدود شعاع التحكم
control_constraints = og.constraints.BallInf(None, 9.4)


u_seq = cs.MX.sym("u", N)  # sequence of all u's
x0 = cs.MX.sym("x0", nx)   # initial state


x_t = x0
total_cost = 0
F1 = []
for t in range(0, N):
    total_cost += stage_cost(x_t, u_seq[t])  # update cost
    x_t = dynamics_dt(x_t, u_seq[t])         # update state
    F1 = cs.vertcat(F1, x_t[1])              # state constraint   
    
    
total_cost += terminal_cost(x_t)  # terminal cost


#################################################
# code generation
problem = og.builder.Problem(u_seq, x0, total_cost)  \
            .with_constraints(control_constraints)   \
            .with_aug_lagrangian_constraints(F1, state_constraints)
            

build_config = og.config.BuildConfiguration()  \
    .with_build_directory("python_build")      \
    .with_tcp_interface_config()

meta = og.config.OptimizerMeta().with_optimizer_name("cart_and_pole")

solver_config = og.config.SolverConfiguration()\
    .with_tolerance(1e-4)\
    .with_initial_tolerance(1e-4)\
    .with_max_duration_micros(100000)

builder = og.builder.OpEnOptimizerBuilder(problem, meta,
                                          build_config, solver_config)
builder.build()
################################################



################################################
# Create a TCP connection manager
mng = og.tcp.OptimizerTcpManager("python_build/cart_and_pole")

# Start the TCP server
mng.start()

# Run simulations
x_state_0 = [0.1, 0.0, 3.14, 0.0]
simulation_steps = 300

state_sequence = x_state_0
input_sequence = []

x = x_state_0
for k in range(simulation_steps):
    solver_status = mng.call(x)
    us = solver_status['solution']
    u = us[0]
    x_next = dynamics_dt(x, u)
    state_sequence += x_next
    input_sequence += [u]
    x = x_next

# Thanks TCP server; we won't be needing you any more
mng.kill()
################################################

# تحويل عناصر state_sequence إلى أرقام عادية
state_sequence_values = []
for state in state_sequence:
    if isinstance(state, (cs.MX, cs.DM)):  # إذا كان العنصر من نوع MX أو DM
        state_sequence_values.append(float(state))  # تحويله إلى رقم عادي
    else:
        state_sequence_values.append(state)

# تأكيد أن جميع العناصر قد تم تحويلها إلى أرقام عادية
# print(state_sequence_values)
print(len(input_sequence))


time = np.arange(0, sampling_time*simulation_steps, sampling_time)

plt.plot(time, state_sequence_values[0:4*simulation_steps:4], '-', label="position")
plt.plot(time, state_sequence_values[1:4*simulation_steps:4], '-', label="cart-velocity")
plt.plot(time, state_sequence_values[2:4*simulation_steps:4], '-', label="angle")
plt.plot(time, state_sequence_values[3:4*simulation_steps:4], '-', label="pendulum-velocity")
plt.plot(time, input_sequence, '-', label="Control Input (Force)")
plt.grid()
plt.ylabel('states')
plt.xlabel('Time')
plt.legend(bbox_to_anchor=(0.7, 0.85), loc='upper left', borderaxespad=0.)
plt.show()