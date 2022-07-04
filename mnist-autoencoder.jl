include("core.jl")

using Random: shuffle
using MLDatasets

train_n = 60000
train_x = reshape(MNIST.traintensor(Float64), 784, :)
train_y = Float64.(0:9 .== transpose(MNIST.trainlabels()))
test_n = 10000
test_x = reshape(MNIST.testtensor(Float64), 784, :)
test_y = Float64.(0:9 .== transpose(MNIST.testlabels()))

encoder = LinearLayer() ⋅ ShiftLayer() ⋅ ScalarActivationLayer(sigmoid) ⋅ LinearLayer() ⋅ ShiftLayer() ⋅ ScalarActivationLayer(sigmoid)
decoder = LinearLayer() ⋅ ShiftLayer() ⋅ ScalarActivationLayer(sigmoid) ⋅ LinearLayer() ⋅ ShiftLayer() ⋅ ScalarActivationLayer(sigmoid)

encoder_parameters = (rand(256, 784) / 784, rand(256), nothing, rand(16, 256) / 256, rand(16), nothing)
decoder_parameters = (rand(256, 16) / 16, rand(256), nothing, rand(784, 256) / 256, rand(784), nothing)

train_model = (GradientDescentOptimizer() ↓ encoder) ⋅ (GradientDescentOptimizer() ↓ decoder) ⋅ ScalarLoss(mse) ⋅ ConstantLearningRate(0.01)

function train(train_model, encoder_parameters, decoder_parameters, input)
    ((encoder_parameters, decoder_parameters, _, _), _) = train_model((encoder_parameters, decoder_parameters, input, nothing), input, nothing)
    return encoder_parameters, decoder_parameters
end

epochs = 5
for epoch in 1:epochs
    @show epoch

    global encoder_parameters, decoder_parameters
    for n in shuffle(1:60000)
        @views encoder_parameters, decoder_parameters = train(train_model, encoder_parameters, decoder_parameters, train_x[:, n])
    end
end

test_model = encoder ⋅ decoder ⋅ ScalarLoss(mse)

function test(test_model, encoder_parameters, decoder_parameters, input)
    loss = test_model((encoder_parameters..., decoder_parameters..., input), input)
    return loss
end

loss = mean(test(test_model, encoder_parameters, decoder_parameters, test_x[:, n]) for n in 1:10000)

@show loss

using ImageCore

n = 1
res = (encoder ⋅ decoder)((encoder_parameters..., decoder_parameters...), test_x[:, n])
MNIST.convert2image(reshape(res, 28, 28))
MNIST.convert2image(test_x[:, n])
