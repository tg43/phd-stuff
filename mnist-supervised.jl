include("core.jl")

using Random: shuffle
using MLDatasets

train_n = 60000
train_x = reshape(MNIST.traintensor(Float64), 784, :)
train_y = Float64.(0:9 .== transpose(MNIST.trainlabels()))
test_n = 10000
test_x = reshape(MNIST.testtensor(Float64), 784, :)
test_y = Float64.(0:9 .== transpose(MNIST.testlabels()))

model = LinearLayer() ⋅ ShiftLayer() ⋅ ScalarActivationLayer(sigmoid) ⋅ LinearLayer() ⋅ ShiftLayer() ⋅ ScalarActivationLayer(sigmoid)
parameters = (rand(100, 784) / 784, rand(100), nothing, rand(10, 100) / 100, rand(10), nothing)  # TODO: Parameter Initalisation (???)

train_model = (GradientDescentOptimizer() ↓ model) ⋅ ScalarLoss(mse) ⋅ ConstantLearningRate(0.01)
# momentums = (zeros(100, 784), zeros(100), nothing, zeros(10, 100), zeros(10), nothing)

# function train(train_model, parameters, momentums, input, output)
#     (((momentums, parameters), _, _), _) = train_model(((momentums, parameters), output, nothing), input, nothing)
#     return parameters, momentums
# end

function train(train_model, parameters, input, output)
    ((parameters, _, _), _) = train_model((parameters, output, nothing), input, nothing)
    return parameters
end

epochs = 10
for epoch in 1:epochs
    @show epoch

    global parameters, momentums
    for n in shuffle(1:60000)
        # @views parameters, momentums = train(train_model, parameters, momentums, train_x[:, n], train_y[:, n])
        @views parameters = train(train_model, parameters, train_x[:, n], train_y[:, n])
    end
end

function test(model, parameters, input, output)
    predicted = model(parameters, input)
    return Int(argmax(predicted) == argmax(output))
end

acc = mean(test(model, parameters, test_x[:, n], test_y[:, n]) for n in 1:test_n)

@show acc
