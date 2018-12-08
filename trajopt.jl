using Plots
using LinearAlgebra
using Random
using JLD2, FileIO

include("./acrobot.jl")

# Global vars
r = .001
k = 2
eps = 1e-8

# Finite difference to get gradients of dynamics with respect
# to either states or controls
function finiteDiff(x, u, diffvar)
    global eps
    if diffvar == "x"   # Diff wrt to x
        diff = zeros(4, 4)
        for i =1:4
            step = zeros(4)
            step[i] = eps
            leftX = rk4(x - step, u)
            rightX = rk4(x + step, u)
            diff[:, i] = (rightX - leftX) ./ (2*eps)
        end
        return diff
    elseif diffvar == "u"
        leftU = rk4(x, u - eps)
        rightU = rk4(x, u + eps)
        return (rightU - leftU) ./ (2*eps)
    else
        print("Error, invalid differentiation variable")
        return 0
    end
end

# Derivative of cost w.r.t. to control
function lu(x, u)
    return r*u
end

# Derivative of cost w.r.t to state
function true_lx(x, u)
    estuff = exp(k*cos(x[1]) + k*cos(x[2]) - 2*k)
    return [-k*sin(x[1])*estuff, -k*sin(x[2])*estuff, 0, 0]
end

# Cost function
function true_cost(x, u)
    return r/2*u^2+1-exp(k*cos(x[1])+k*cos(x[2])-2*k)
end

# Compute total cost of a trajectory
function total_cost(pos, u)
    total_cost = 0
    for j = 1:length(u)
        total_cost += true_cost(pos[:, j], u[j])
    end
    return total_cost
end

# Get lambdas to compute gradient
function getlambdas(x, u)
    T = length(u)
    lambdas = zeros(4, T+1)
    lambdas[:, T+1] = true_lx(x[:, T+1], 0)
    for i = T:-1:1
        costdiff = true_lx(x[:, i], u[i])
        next = costdiff + transpose(finiteDiff(x[:, i], u[i], "x")) * lambdas[:, i+1]
        lambdas[:, i] = next
    end
    return lambdas
end

# Compute gradient of total cost w.r.t to controls
function getgrads(x, u)
    T = length(u)
    grads = zeros(T)
    lambdas = getlambdas(x, u)
    for i = 1:T
        fxlambda = transpose(finiteDiff(x[:, i], u[i], "u")) * lambdas[:, i+1]
        grads[i] = lu(x[:, i], u[i]) + fxlambda
    end
    return grads
end

# Optimize a trajectory given an initial state and initial guess at optimal
# control sequence.
function trajopt(xinit, uinit, T, niter, max_step)
    u = copy(uinit)
    costs = zeros(0)
    min_step = 1e-4
    for i = 1:niter
        curr_step = max_step
        pos = forward(xinit, u)
        curr_cost = total_cost(pos, u)
        append!(costs, curr_cost)
        println("Iteration: "*string(i)*"\tTotal Cost: "*string(curr_cost))
        if i > 1 && costs[i] == costs[i-1]
            break
        end
        grads = getgrads(pos, u)
        do_adapt = true
        newu = u + curr_step*grads
        rollout = forward(xinit, newu)
        roll_cost = total_cost(rollout, newu)
         if roll_cost > curr_cost
           break
         end
        if do_adapt
            while (roll_cost > curr_cost)
                curr_step = curr_step / 2
                newu = u + curr_step*grads
                rollout = forward(xinit, newu)
                roll_cost = total_cost(rollout, newu)
            end
        end
        u = newu
    end
    return u, costs
end

gr()
println("start")
T = 3
step = 1e-2
niter = 500

xinit = [0, 0, 0, 0] + .05*rand(4,1)
xinit = [-0.0531, 0.1175, -0.0308, 0.0374]
# println("xinit: ", xinit)

uinit = zeros(Int(T/dt))
# uinit = 4*rand(Int(T/dt)) .- 2
@time u, costs = trajopt(xinit, uinit, T, niter, step)
# finalc = Inf
# fcosts = zeros(0)
# for i = 1:10
#     uinit = 4*rand(Int(T/dt)) .- 2
#     global finalc
#     @time u, costs = trajopt(xinit, uinit, T, niter, step)
#     if minimum(costs) < finalc
#         println("Saved new traj")
#         finalc = minimum(costs)
#         fcosts = copy(costs)
#     end
# end
# println("Final cost: ", finalc)

plot(costs, xlabel="Iteration", ylabel="Cost", title="Cost Curve of Gradient Descent with Adaptive Step Size")
savefig("adaptstep.png")

pos = forward(xinit, u)
println("animating")
anim = @animate for i = 1:3:Int(T/dt) # Only plot every third frame
    coord = cartcoord(pos[:, i])
    plot([0, coord[1], coord[3]], [0, coord[2], coord[4]], xlim=(-2, 2), ylim=(-2,2))
end
gif(anim, "./adapt.gif")
