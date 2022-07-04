# This is a first implementation of "Categorical Foundations of Gradient-Based Learning", a python implementation is provided by the authors at https://github.com/statusfailed/numeric-optics-python

# This file contains the core components.
# "mnist-supervised.jl" and "mnist-autoencoder.jl" apply this to the MNIST dataset.

using Statistics: mean

# ---------------------------------------------------------------------------------------- #
# Lens                                                                                     #
# ---------------------------------------------------------------------------------------- #

abstract type Lens end

(lens::Lens)(a) = forward(lens, a)
(lens::Lens)(a, b′) = backward(lens, a, b′)

struct IdentityLens <: Lens end
forward(_::IdentityLens, a) = a
backward(_::IdentityLens, _, b′) = b′
reverse(_::IdentityLens, b) = b
id = IdentityLens()

struct ComposedLens{N, T <: NTuple{N, Lens}} <: Lens
    lenses::T

    function ComposedLens(lenses...)
        return new{length(lenses), typeof(lenses)}(lenses)
    end
end

⋅(inner::Lens, outer::Lens) = ComposedLens(inner, outer)
⋅(inner::Lens, outer::ComposedLens) = ComposedLens(inner, outer.lenses...)
⋅(inner::ComposedLens, outer::Lens) = ComposedLens(inner.lenses..., outer)
⋅(inner::ComposedLens, outer::ComposedLens) = ComposedLens(inner.lenses..., outer.lenses...)

Base.:∘(outer::Lens, inner::Lens) = inner ⋅ outer

_forward(_::Tuple{}, a) = a
function _forward((head, tail...)::NTuple{N, Lens}, a) where N
    return _forward(tail, head(a))
end

function forward(lens::ComposedLens, a)
    return _forward(lens.lenses, a)
end

_backward(_::Tuple{}, _, b′) = b′
function _backward((head, tail...)::NTuple{N, Lens}, a, b′) where N
    return head(a, _backward(tail, head(a), b′))
end

function backward(lens::ComposedLens, a, b′)
    return _backward(lens.lenses, a, b′)
end

struct ProductLens{N, T <: NTuple{N, Lens}} <: Lens
    lenses::T

    function ProductLens(lenses...)
        return new{length(lenses), typeof(lenses)}(lenses)
    end
end

×(left::Lens, right::Lens) = ProductLens(left, right)
×(left::Lens, right::ProductLens) = ProductLens(left, right.lenses...)
×(left::ProductLens, right::Lens) = ProductLens(left.lenses..., right)
×(left::ProductLens, right::ProductLens) = ProductLens(left.lenses..., right.lenses...)

function forward(lens::ProductLens{N}, a::NTuple{N, Any}) where N
    N::Int

    if @generated
        return quote
            Base.@nexprs $N i -> b_i = lens.lenses[i](a[i])
            Base.@ntuple $N b
        end

    else
        return Tuple(lens(a) for (lens, a) in zip(lens.lenses, a))
    end
end

function backward(lens::ProductLens{N}, a::NTuple{N, Any}, b′::NTuple{N, Any}) where N
    N::Int

    if @generated
        return quote
            Base.@nexprs $N i -> a′_i = lens.lenses[i](a[i], b′[i])
            Base.@ntuple $N a′
        end

    else
        return Tuple(lens(a, b′) for (lens, a, b′) in zip(lens.lenses, a, b′))
    end
end

# ---------------------------------------------------------------------------------------- #
# ParaLens                                                                                 #
# ---------------------------------------------------------------------------------------- #

abstract type ParaLens end

(lens::ParaLens)(p, a) = forward(lens, p, a)
(lens::ParaLens)(p, a, b′) = backward(lens, p, a, b′)

struct LiftedLens{T <: Lens} <: ParaLens
    lens::T
end

forward(lens::LiftedLens, _::Nothing, a) = lens.lens(a)
backward(lens::LiftedLens, _::Nothing, a, b′) = nothing, lens.lens(a, b′)
reverse(lens::LiftedLens, _::Nothing, b) = reverse(lens.lens, b)

struct ComposedParaLens{N, T <: NTuple{N, ParaLens}} <: ParaLens
    lenses::T

    function ComposedParaLens(lenses...)
        return new{length(lenses), typeof(lenses)}(lenses)
    end
end

⋅(inner::ParaLens, outer::ParaLens) = ComposedParaLens(inner, outer)
⋅(inner::ParaLens, outer::ComposedParaLens) = ComposedParaLens(inner, outer.lenses...)
⋅(inner::ComposedParaLens, outer::ParaLens) = ComposedParaLens(inner.lenses..., outer)
⋅(inner::ComposedParaLens, outer::ComposedParaLens) = ComposedParaLens(inner.lenses..., outer.lenses...)

Base.:∘(outer::ParaLens, inner::ParaLens) = inner ⋅ outer

_forward(_::Tuple{}, _::Tuple{}, a) = a
function _forward((head, tail...)::NTuple{N, ParaLens}, (p, q...)::NTuple{N, Any}, a) where N
    return _forward(tail, q, head(p, a))
end

function forward(lens::ComposedParaLens{N}, p::NTuple{N, Any}, a) where N
    return _forward(lens.lenses, p, a)
end

_backward(_::Tuple{}, _::Tuple{}, _, c′) = (), c′
function _backward((head, tail...)::NTuple{N, ParaLens}, (p, q...)::NTuple{N, Any}, a, c′) where N
    q′, b′ = _backward(tail, q, head(p, a), c′)
    p′, a′ = head(p, a, b′)
    return (p′, q′...), a′
end

function backward(lens::ComposedParaLens{N}, p::NTuple{N, Any}, a, b′) where N
    return _backward(lens.lenses, p, a, b′)
end

struct StackedParaLens{S <: Union{Lens, ParaLens}, T <: ParaLens} <: ParaLens
    stacked::S
    primary::T
end

↓(stacked, primary) = StackedParaLens(stacked, primary)

function forward(lens::StackedParaLens{T}, p, a) where T <: Lens
    return lens.primary(lens.stacked(p), a)
end

function backward(lens::StackedParaLens{T}, p, a, b′) where T <: Lens
    q′, a′ = lens.primary(lens.stacked(p), a, b′)
    p′ = lens.stacked(p, q′)
    return p′, a′
end

function forward(lens::StackedParaLens{T}, (p, q)::Tuple{Any, Any}, a) where T <: ParaLens
    return lens.primary(lens.stacked(p, q), a)
end

function backward(lens::StackedParaLens{T}, (p, q)::Tuple{Any, Any}, a, b′) where T <: ParaLens
    r′, a′ = lens.primary(lens.stacked(p, q), a, b′)
    p′, q′ = lens.stacked(p, q, r′)
    return (p′, q′), a′
end

struct ProductParaLens{N, T <: NTuple{N, ParaLens}} <: ParaLens
    lenses::T

    function ProductParaLens(lenses...)
        return new{length(lenses), typeof(lenses)}(lenses)
    end
end

×(left::ParaLens, right::ParaLens) = ProductParaLens(left, right)
×(left::ParaLens, right::ProductParaLens) = ProductParaLens(left, right.lenses...)
×(left::ProductParaLens, right::ParaLens) = ProductParaLens(left.lenses..., right)
×(left::ProductParaLens, right::ProductParaLens) = ProductParaLens(left.lenses..., right.lenses...)

function forward(lens::ProductParaLens{N}, p::NTuple{N, Any}, a::NTuple{N, Any}) where N
    N::Int

    if @generated
        return quote
            Base.@nexprs $N i -> b_i = lens.lenses[i](p[i], a[i])
            Base.@ntuple $N b
        end

    else
        return Tuple(lens(p, a) for (lens, p, a) in zip(lens.lenses, p, a))
    end
end

function backward(lens::ProductParaLens{N}, p::NTuple{N, Any}, a::NTuple{N, Any}, b′::NTuple{N, Any}) where N
    N::Int

    if @generated
        return quote
            Base.@nexprs $N i -> (p′_i, a′_i) = lens.lenses[i](p[i], a[i], b′[i])
            (Base.@ntuple $N p′), (Base.@ntuple $N a′)
        end

    else
        result = Tuple(lens(p, a, b′) for (lens, p, a, b′) in zip(lens.lenses, p, a, b′))
        return first.(result), last.(result)
    end
end

# ---------------------------------------------------------------------------------------- #
# Initialisation                                                                           #
# ---------------------------------------------------------------------------------------- #



# ---------------------------------------------------------------------------------------- #
# Components                                                                               #
# ---------------------------------------------------------------------------------------- #

struct LinearLayer <: ParaLens end

function forward(_::LinearLayer, p::AbstractMatrix{T}, a::AbstractVector{T}) where T
    return p * a
end

function backward(l_ens::LinearLayer, p::AbstractMatrix{T}, a::AbstractVector{T}, b′::AbstractVector{T}) where T
    p′ = b′ * a'
    a′ = p' * b′
    return p′, a′
end

struct ShiftLayer <: ParaLens end

function forward(_::ShiftLayer, p::AbstractVector{T}, a::AbstractVector{T}) where T
    return a + p
end

function backward(_::ShiftLayer, p::AbstractVector{T}, a::AbstractVector{T}, b′::AbstractVector{T}) where T
    return b′, b′
end

struct ActivationFunction{F, F′}
    f::F
    f′::F′
end

identity = ActivationFunction(x -> x, x -> one(x))
sigmoid = ActivationFunction(x -> 1 / (1 + exp(-x)), x -> exp(-x) / (1 + exp(-x))^2)
relu = ActivationFunction(x -> x > 0 ? x : zero(x), x -> x > 0 ? one(x) : zero(x))
leakyrelu = ActivationFunction(x -> x > 0 ? x : 0.01 * x, x -> x > 0 ? 1.0 : 0.01)

struct ScalarActivationLayer{F <: ActivationFunction} <: ParaLens
    f::F
end

function forward(lens::ScalarActivationLayer, _::Nothing, a::AbstractVector)
    return lens.f.f.(a)
end

function backward(lens::ScalarActivationLayer, _::Nothing, a::AbstractVector, b′::AbstractVector)
    return nothing, b′ .* lens.f.f′.(a)
end

struct GradientDescentOptimizer <: Lens end
forward(_::GradientDescentOptimizer, a) = a
backward(_::GradientDescentOptimizer, _::Nothing, _::Nothing) = nothing
backward(_::GradientDescentOptimizer, a, b′) = a - b′
backward(lens::GradientDescentOptimizer, a::NTuple{N, Any}, b′::NTuple{N, Any}) where N = lens.(a, b′)

struct GradientAscentOptimizer <: Lens end
forward(_::GradientAscentOptimizer, a) = a
backward(_::GradientAscentOptimizer, _::Nothing, _::Nothing) = nothing
backward(_::GradientAscentOptimizer, a, b′) = a + b′
backward(lens::GradientAscentOptimizer, a::NTuple{N, Any}, b′::NTuple{N, Any}) where N = lens.(a, b′)

struct MomentumOptimizer{T} <: ParaLens
    β::T
end
forward(_::MomentumOptimizer, p, a) = a
backward(_::MomentumOptimizer, _::Nothing, _::Nothing, _::Nothing) = nothing, nothing
function backward(lens::MomentumOptimizer, p, a, b′)
    p′ = lens.β * p + (1 - lens.β) * b′
    a′ = a - p′
    return p′, a′
end
function backward(lens::MomentumOptimizer, p::NTuple{N, Any}, a::NTuple{N, Any}, b′::NTuple{N, Any}) where N
    result = lens.(p, a, b′)
    return first.(result), last.(result)
end

struct LossFunction{F <: Function, Fˣ <: Function, Fʸ <: Function}
    f::F
    fˣ::Fˣ
    fʸ::Fʸ
end

mse = LossFunction((x, y) -> (x - y)^2, (x, y) -> 2 * (x - y), (x, y) -> 2 * (y - x))

struct ScalarLoss{F <: LossFunction} <: ParaLens
    f::F
end

function forward(lens::ScalarLoss, p::AbstractVector{T}, a::AbstractVector{T}) where T
    return mean(lens.f.f.(p, a))
end

function backward(lens::ScalarLoss, p::AbstractVector{T}, a::AbstractVector{T}, b′::T) where T
    return b′ * lens.f.fˣ.(p, a), b′ * lens.f.fʸ.(p, a)
end

struct ConstantLearningRate{T} <: ParaLens
    ϵ::T
end

forward(lens::ConstantLearningRate{T}, _::Nothing, _::T) where T = nothing
backward(lens::ConstantLearningRate{T}, _::Nothing, _::T, _::Nothing) where T = nothing, lens.ϵ
