using Plots
# using ControlSystems
using LinearAlgebra

# Global vars
r = 1
k = 2
eps = 1e-6
damping = 0.1
gravity = 9.8
dt = 0.01
# xinit = zeros(4,1)+ones(4)*.01#0.01*randn(4,1)
xinit = [-0.0534, -0.0405, -0.1472, 0.0719]


function rk4(state, u)
    massPos = state[1:2]
    massVel = state[3:4]
    l0 = dt * xdot(state, u)[3:4]
    k0 = dt * massVel;
    l1 = dt * xdot(vcat(massPos + k0*0.5, massVel + l0*0.5), u)[3:4]
    k1 = dt * (massVel + k0*0.5);
    l2 = dt * xdot(vcat(massPos + k1*0.5, massVel + l1*0.5), u)[3:4]
    k2 = dt * (massVel + l1*0.5);
    l3 = dt * xdot(vcat(massPos + k2, massVel + l2), u)[3:4]
    k3 = dt * (massVel + l2);
    massPosNext = massPos + 1.0/6 * (k0 + 2*k1 + 2*k2 + k3);
    massVelNext = massVel + 1.0/6 * (l0 + 2*l1 + 2*l2 + l3);
    return vcat(massPosNext, massVelNext)
end

function int_lambda(lambda, x, u)
    lam_vel = lambdadot(lambda, x, u)
    return lambda + dt*lam_vel
end

function cartcoord(state)
    x1 = cos(state[1]+pi/2);
    y1 = sin(state[1]+pi/2);
    x2 = x1 + cos(state[1]+state[2]+pi/2);
    y2 = y1 + sin(state[1]+state[2]+pi/2);
    return [x1, y1, x2, y2]
end

function xdot(x, u)
    # damping = 0.1
    # gravity = 9.8
    # inertia matrix
    M = vcat(transpose([3 + 2*cos(x[2]), 1+cos(x[2])]), transpose([1+cos(x[2]), 1]))

    # Coriolis, centripetal and graviational forces
    c1 = x[4]*(2*x[3]+x[4])*sin(x[2]) + 2*gravity*sin(x[1]) + gravity*sin(x[1]+x[2])
    c2 = -x[3]^2*sin(x[2]) + gravity*sin(x[1]+x[2])
    
    # passive dynamics
    a = vcat(x[3:4], M \ [c1-damping*x[3], c2-damping*x[4]])

    # control gain; use B = [0;0; inv(M)] to allow control of both joints
    Mback = M\[0,1]
    B = [0,0, Mback[1], Mback[2]]

    xdot = a + B*u

    return xdot
end

function xdotlin(x, u)
    A = hcat([0, 0, 1, 0], [0, 0, 0, 1], [gravity, -gravity, 0, 0], [-gravity, 3*gravity, 0, 0])
    A = transpose(A)
    B = [0, 0, -2, 5]
    return A*x + B*u
end

function lambdadot(lambda, x, u)
    return true_lx(x, u) + transpose(finiteDiff(x, u, "x")) * lambda
end

function finiteDiff(x, u, diffvar)
    # eps =   
    if diffvar == "x"   # Diff wrt to x
        diff = zeros(4, 4)
        for i =1:4
            step = zeros(4)
            step[i] = eps
            leftX = rk4(x - step, u)
            rightX = rk4(x + step, u)
            diff[:, i] = (rightX - leftX) / 2*eps
        end
        return diff
    elseif diffvar == "u"
        step = 1e-3
        leftU = rk4(x, u - eps)
        rightU = rk4(x, u + eps)
        return (rightU - leftU) / 2*eps
    else
        print("Error, invalid differentiation variable")
        return 0
    end
end

function wrapstate!(state)
    while state[1] > pi
        state[1] -= 2*pi
    end
    while state[1] < -pi
        state[1] += 2*pi
    end
    while state[2] > pi
        state[2] -= 2*pi
    end
    while state[2] < -pi
        state[2] += 2*pi
    end
end

function forward(xinit, u)
    T = length(u) + 1
    pos = zeros(4, T)
    pos[:,1] = xinit
    G = [421.7063, 140.6532, 173.1010, 65.7230]
    for i = 2:T
        uLQR = transpose(G)*pos[:, i-1]
        next = rk4(pos[:, i-1], u[i-1])
        wrapstate!(next)
        pos[:, i] = next
    end
    return pos
end

function lu(x, u)
    r = 1
    return r*u
end

function lx(x, u)
    k = 2
    return 1/2*k^2*[2*x[1], 2*x[2], 0, 0] + k*[1, 1, 0, 0]
end

function true_lx(x, u)
    k = 2
    estuff = exp(k*cos(x[1]) + k*cos(x[2]) - 2*k)
    return [-k*sin(x[1])*estuff, -k*sin(x[2])*estuff, 0, 0]
end

function cost(x, u)
    r = 1
    k = 2
    return r/2*u^2 + 1/2*k^2*(x[1]^2+x[2]^2)+k*(x[1]+x[2]) + 1
end

function true_cost(x, u)
    r = 1
    k = 2
    return r/2*u^2+1-exp(k*cos(x[1])+k*cos(x[2])-2*k)
end

function cost_func(u)
    pos = forward(xinit, u)
    return total_cost(pos, u)
end

function total_cost(pos, u)
    total_cost = 0
    for j = 1:length(u)
        total_cost = true_cost(pos[:, j], u[j])
    end
    return total_cost
end

function getlambdas(x, u)
    T = length(u)
    lambdas = zeros(4, T+1)
    lambdas[:, T+1] = true_lx(x[:, T+1], 0)
    for i in T:-1:2
        costdiff = true_lx(x[:, i], u[i])
        next = costdiff + transpose(finiteDiff(x[:, i], u[i], "x")) * lambdas[:, i+1]
        lambdas[:, i-1] = next #costdiff + transpose(finiteDiff(x[:, i-1], u[i-1], "x")) * lambdas[:, i]
    end
    return lambdas
end

function getgrads(x, u)
    T = length(u)
    grads = zeros(T)
    lambdas = getlambdas(x, u)
    # lambdas = zeros(4, T+1)
    # lambdas[:, T+1] = true_lx(x[:, T+1], 0)
    # lambdas[:, T] = int_lambda(lambdas[:, T+1], x[:, T+1], 0)
    # for i = T-1:-1:1
        # lambdas[:, i] = int_lambda(lambdas[:, i+1], x[:, i+1], u[i+1])
    # end
    for i in 1:T
        fxlambda = transpose(finiteDiff(x[:, i], u[i], "u")) * lambdas[:, i+1]
        # print(fxlambda)
        grads[i] = lu(x[:, i], u[i]) + fxlambda
    end
    return grads
end

function uGrad!(G, u)
    x = forward(xinit, u)
    lambdas = zeros(4, T+1)
    lambdas[:, T+1] = lx(x[:, T+1], 0)
    for i = T:-1:1
        lambdas[:, i] = int_lambda(lambdas[:, i+1], x[:, i], u[i])
    end
    for i in 1:T
        G[i] = lu(x[:, i], u[i]) + transpose(finiteDiff(x[:, i], u[i], "u")) * lambdas[:, i+1]
    end
end


function step_search(x, u, curr_cost, grad, step)
    newu = u + step*grad
    rollout = forward(x, newu)
    roll_cost = total_cost(rollout, newu)
    step_incr = step/10
    curr_step = step
    while roll_cost > curr_cost
        curr_step -= step_incr
        newu = u + curr_step*grad
        rollout = forward(x, newu)
        roll_cost = total_cost(rollout, newu)
    end
    return newu
end

function trajopt(xinit, uinit, T, niter, max_step)
    u = copy(uinit)
    costs = zeros(niter)
    min_step = 1e-4
    for i = 1:niter
        curr_step = max_step
        # if i > 180
        #     curr_step = 1e-5
        # end
        pos = forward(xinit, u)
        curr_cost = total_cost(pos, u)
        costs[i] = curr_cost
        println("Iteration: "*string(i)*"\tTotal Cost: "*string(curr_cost))
        grads = getgrads(pos, u)
        println("Grads: ", LinearAlgebra.norm(grads))
        # Adaptive step size:
        # newu = step_search(xinit, u, curr_cost, grads, curr_step)
        # while newu == u
        #     curr_step = curr_step * .1
        #     newu = step_search(xinit, u, curr_cost, grads, curr_step)
        # end
        # println(curr_step)
        newu = u - curr_step*grads
        # rollout = forward(xinit, newu)
        # roll_cost = total_cost(rollout, newu)
        # while (roll_cost > curr_cost)# && curr_step > 1e-6)
        #     curr_step = curr_step / 2
        #     newu = u + curr_step*grads
        #     rollout = forward(xinit, newu)
        #     roll_cost = total_cost(rollout, newu)
        # end
        # println(curr_step)
        u = newu
    end
    return u, costs
end

function accel_trajopt(xinit, niter, max_step)
    prev_a = 1
    curr_a = .5*(1+sqrt(4+1))
    # prev_u = zeros(Int(T/dt))
    curr_u = zeros(Int(T/dt))
    curr_y = zeros(Int(T/dt))
    costs = zeros(niter)
    for i = 1:niter
        curr_traj = forward(xinit, curr_u)
        curr_cost = total_cost(curr_traj, curr_u)
        costs[i] = curr_cost
        println("Iteration: "*string(i)*"\tTotal Cost: "*string(curr_cost))
        next_a = .5*(1+sqrt(4*curr_a^2+1))
        t = (1-curr_a) / next_a
        # pos = forward(xinit, y)
        grads = getgrads(curr_traj, curr_u)
        next_y = curr_u + max_step*grads
        new_u = (1-t)*next_y + t*curr_y
        
        # new_u = y + max_step*grads
        curr_a = next_a
        # prev_u = curr_u
        curr_u = new_u
    end
    return curr_u, costs
end

# TODO: Clamp controls. properly wrap angles, not just subtract pi in case wraps around more than once in a time step (but shouldn't happen
# if limit controls)
# BISECTION line search for grad step!!


gr()
println("start")
T = 5
step = 1e0
niter = 100

# A = hcat([0, 0, 1, 0], [0, 0, 0, 1], [gravity, -gravity, 0, 0], [-gravity, 3*gravity, 0, 0])
# A = transpose(A)
# println(size(A))
# B = [0, 0, -2, 5]
# Q = Matrix{Float64}(I, 4, 4)
# R = Matrix{Float64}(I, 4, 4)
# println(transpose(B)*R*B)
# X = ControlSystems.care(A, B, Q, R)
# G = transpose(B)*X
# println(G)
uinit = zeros(Int(T/dt))
# optimize(cost_func, uGrad!, uinit)
@time u, costs = trajopt(xinit, uinit, T, niter, step)
# @time u, costs = accel_trajopt(xinit, niter, step)

plot(costs)
savefig("truecost.png")

# u = zeros(Int(T/dt))
pos = forward(xinit, u)
# total_cost = 0
# for j = 1:length(u)
#     total_cost = cost(pos[:, j], u[j])
# end
println("animating")
anim = @animate for i = 1:3:Int(T/dt) # Only plot every third frame
    coord = cartcoord(pos[:, i])
    plot([0, coord[1], coord[3]], [0, coord[2], coord[4]], xlim=(-2, 2), ylim=(-2,2))
    # frame(anim)
end
gif(anim, "./testopt.gif")
