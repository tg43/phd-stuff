using LinearAlgebra
using Random

using BenchmarkTools
using ProgressBars
using MLDatasets


abstract type Learner end

(learner::Learner)(input::Vector{Float64}) = apply(learner, input)

function Base.:show(io::IO, learner::Learner)
    name = learner.name
    input_size = learner.input_size
    output_size = learner.output_size
    print(io, "$name($input_size -> $output_size)")
end

outer(x::Vector{Float64}, y::Vector{Float64}) = x * y'
ε = 0.01

struct LinearLearner <: Learner
    input_size::Int
    output_size::Int

    name::String
    weights::Matrix{Float64}
    biases::Vector{Float64}

    function LinearLearner(input_size, output_size, weights, biases)
        @assert size(weights) == (output_size, input_size)
        @assert size(biases) == (output_size,)

        return new(input_size, output_size, "Linear", weights, biases)
    end
end

function LinearLearner(weights::Matrix{Float64}, biases::Vector{Float64})
    output_size, input_size = size(weights)
    return LinearLearner(input_size, output_size, weights, biases)
end

function LinearLearner(input_size::Int, output_size::Int)
    @assert input_size > 0 && output_size > 0

    weights = rand(output_size, input_size)
    biases = rand(output_size)
    return LinearLearner(input_size, output_size, weights, biases)
end

# Implementation Function: I: P × A -> B
function apply(learner::LinearLearner, input::Vector{Float64})
    @assert size(input) == (learner.input_size,)
    return learner.weights * input + learner.biases
end

# Update & Request Function: U: P × A × B -> P × A
function update(learner::LinearLearner, input::Vector{Float64}, expected::Vector{Float64})
    @assert size(input) == (learner.input_size,)
    @assert size(expected) == (learner.output_size,)

    predicted = learner(input)

    # Update
    weights = learner.weights - ε * outer(predicted - expected, input)
    biases = learner.biases - ε * (predicted - expected)
    updated = LinearLearner(weights, biases)

    # Request
    requested = input - transpose(learner.weights) * (predicted - expected)

    return updated, requested
end

struct ActivationLearner <: Learner
    input_size::Int
    output_size::Int

    name::String
    f::Function
    Df::Function

    function ActivationLearner(input_size, output_size, name, f, Df)
        @assert input_size == output_size
        return new(input_size, output_size, name, f, Df)
    end
end

function Sigmoid(size::Int)
    σ(x::Float64) = 1 / (1 + exp(-x))
    Dσ(x::Float64) = σ(x) * (1 - σ(x))
    return ActivationLearner(size, size, "Sigmoid", σ, Dσ)
end

function ReLU(size::Int)
    f(x::Float64) = x > 0 ? x : 0.0
    Df(x::Float64) = x > 0 ? 1.0 : 0.0
    return ActivationLearner(size, size, "ReLU", f, Df)
end

function LeakyReLU(size::Int)
    f(x::Float64) = x > 0 ? x : 0.01 * x
    Df(x::Float64) = x > 0 ? 1.0 : 0.01
    return ActivationLearner(size, size, "LeakyReLU", f, Df)
end

# Implementation Function: I: P × A -> B
function apply(learner::ActivationLearner, input::Vector{Float64})
    @assert size(input) == (learner.input_size,)
    return learner.f.(input)
end

# Update & Request Function: U: P × A × B -> P × A
function update(learner::ActivationLearner, input::Vector{Float64}, expected::Vector{Float64})
    @assert size(input) == (learner.input_size,)
    @assert size(expected) == (learner.output_size,)

    # Update
    updated = learner

    # Request
    predicted = learner(input)
    requested = input - (predicted - expected) .* learner.Df.(input)

    return updated, requested
end

struct SoftmaxLearner <: Learner
    input_size::Int
    output_size::Int
    name::String

    function SoftmaxLearner(input_size, output_size)
        @assert input_size == output_size
        return new(input_size, output_size, "Softmax")
    end
end

function Softmax(size::Int)
    return SoftmaxLearner(size, size)
end

# Implementation Function: I: P × A -> B
function apply(learner::SoftmaxLearner, input::Vector{Float64})
    @assert size(input) == (learner.input_size,)
    shifted = input .- maximum(input)
    return exp.(shifted) / sum(exp.(shifted))
end

# Update & Request Function: U: P × A × B -> P × A
function update(learner::SoftmaxLearner, input::Vector{Float64}, expected::Vector{Float64})
    @assert size(input) == (learner.input_size,)
    @assert size(expected) == (learner.output_size,)

    # Update
    updated = learner

    predicted = learner(input)
    A = repeat(predicted; inner=(1, learner.output_size))
    b = (predicted - expected) .* predicted
    requested = input - (I - A) * b

    return updated, requested
end

struct CompositeLearner{S <: Learner, T <: Learner} <: Learner
    input_size::Int
    output_size::Int

    right::S
    left::T

    function CompositeLearner(input_size, output_size, right, left)
        @assert input_size == right.input_size
        @assert right.output_size == left.input_size
        @assert left.output_size == output_size
        return new{typeof(right), typeof(left)}(input_size, output_size, right, left)
    end
end

function CompositeLearner(right::Learner, left::Learner)
    input_size = right.input_size
    output_size = left.output_size

    return CompositeLearner(input_size, output_size, right, left)
end

function Base.:∘(left::Learner, right::Learner)
    return CompositeLearner(right, left)
end

function Base.:show(io::IO, learner::CompositeLearner)
    left = learner.left
    right = learner.right
    print(io, "$right => $left")
end

# Implementation Function: I: P × A -> B
function apply(learner::CompositeLearner, input::Vector{Float64})
    @assert size(input) == (learner.input_size,)
    return learner.left(learner.right(input))
end

# (P × A -> B, P × A -> B -> P × A)

# Update & Request Function: U: P × A × B -> P × A
function update(learner::CompositeLearner, input::Vector{Float64}, expected::Vector{Float64})
    @assert size(input) == (learner.input_size,)
    @assert size(expected) == (learner.output_size,)

    # Update
    right_output = learner.right(input)
    left, left_request = update(learner.left, right_output, expected)
    right, right_request = update(learner.right, input, left_request)
    updated = left ∘ right

    # Request
    requested = right_request

    return updated, requested
end

step(learner, data) = update(learner, data.input, data.output) |> first
train(learner, data, epochs) = foldl(step, Iterators.repeat(data, epochs); init=learner)

function main()
    train_x, train_y = MNIST.traindata()
    test_x, test_y = MNIST.testdata()

    learner = LinearLearner(784, 10)

    # Train
    inputs = [Float64.(reshape(train_x[:, :, k], 784)) for k in 1:60000]
    outputs = [Float64.(0:9 .== train_y[k]) for k in 1:60000]
    data = [(input=input, output=output) for (input, output) in zip(inputs, outputs)]
    result = train(learner, data, 200)

    # Test
    inputs = [Float64.(reshape(test_x[:, :, k], 784)) for k in 1:10000]
    outputs = argmax.(result.(inputs)) .- 1

    correct = 0
    for (predicted, expected) in zip(outputs, test_y)
        if predicted == expected
            correct += 1
        end
    end

    println("Accuracy: $(correct / 10000)")
end

#function main()
#    inputs = [rand([0.0, 1.0], 2) for _ in 1:100]
#    outputs = [[Float64(isodd(sum(input)))] for input in inputs]
#
#    learner = Sigmoid(1) ∘ LinearLearner(3, 1) ∘ Sigmoid(3) ∘ LinearLearner(2, 3)
#
#    for _ in 1:100
#        for (input, output) in zip(inputs, outputs)
#            learner = step(learner, (input=input, output=output))
#        end
#    end
#
#    N = 100
#    inputs = [rand([0.0, 1.0], 2) for _ in 1:N]
#    outputs = [[Float64(isodd(sum(input)))] for input in inputs]
#
#   correct = 0
#    for (input, output) in zip(inputs, outputs)
#        if output[1] == round(learner(input)[1])
#            correct += 1
#        end
#    end
#
#    # println("Test Acc: $(correct / N)")
#end

main()
