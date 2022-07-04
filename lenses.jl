abstract type Lens end

function (lens::Lens)(a)
    _, b = forward(lens, a)
    return b
end

function (lens::Lens)(a, b′)
    m, b = forward(lens, a)
    a′ = backward(lens, m, b′)
    return b, a′
end

struct IdentityLens <: Lens end
forward(lens::IdentityLens, a) = nothing, a
backward(lens::IdentityLens, m, b′) = b′
inverse(lens::IdentityLens, b) = b
id = IdentityLens()

struct ComposedLens{N, T <: NTuple{N, Lens}} <: Lens
    lenses::T

    function ComposedLens(lenses...)
        flat = foldl(_composed_lens, lenses; init=())
        return new{length(flat), typeof(flat)}(flat)
    end
end

_composed_lens(xs::Tuple, x::Lens) = (xs..., x)
_composed_lens(xs::Tuple, x::ComposedLens) = (xs..., x.lenses...)

Base.:∘(outer::Lens, inner::Lens) = ComposedLens(inner, outer)
⨟(inner::Lens, outer::Lens) = ComposedLens(inner, outer)

function forward(lens::ComposedLens{N}, a) where N
    if @generated
        return quote
            b_0 = a
            Base.@nexprs $N i -> (m_i, b_i) = forward(lens.lenses[i], b_{i - 1})
            return (Base.@ntuple $N m), $(Symbol("b_", N))
        end

    else
        m = Vector(undef, N)
        for i in 1:N
            m[i], a = forward(lens.lenses[i], a)
        end

        return Tuple(m), a
    end
end

function backward(lens::ComposedLens{N}, m::NTuple{N, Any}, b′) where N
    if @generated
        return quote
            $(Symbol("a′_", N)) = b′
            @nexprs_reverse $N i -> a′_{i - 1} = backward(lens.lenses[i], m[i], a′_i)
            return a′_0
        end

    else
        for i in N:-1:1
            b′ = backward(lens.lenses[i], m[i], b′)
        end

        return b′
    end
end

function inverse(lens::ComposedLens{N}, b) where N
    if @generated
        return quote
            $(Symbol("a_", N)) = b
            @nexprs_reverse $N i -> a_{i - 1} = inverse(lens.lenses[i], a_i)
            return a_0
        end

    else
        for i in N:-1:1
            b = inverse(lens.lenses[i], b)
        end

        return b
    end
end

struct ProductLens{N, T <: NTuple{N, Lens}} <: Lens
    lenses::T

    function ProductLens(lenses...)
        flat = foldl(_product_lens, lenses; init=())
        return new{length(flat), typeof(flat)}(flat)
    end
end

_product_lens(xs::Tuple, x::Lens) = (xs..., x)
_product_lens(xs::Tuple, x::ProductLens) = (xs..., x.lenses...)

×(left::Lens, right::Lens) = ProductLens(left, right)

function forward(lens::ProductLens{N}, a::NTuple{N, Any}) where N
    if @generated
        return quote
            Base.@nexprs $N i -> (m_i, b_i) = forward(lens.lenses[i], a[i])
            return (Base.@ntuple $N m), (Base.@ntuple $N b)
        end

    else
        m, b = Vector(undef, N), Vector(undef, N)
        for i in 1:N
            m[i], b[i] = forward(lens.lenses[i], a[i])
        end

        return Tuple(m), Tuple(b)
    end
end

function backward(lens::ProductLens{N}, m::NTuple{N, Any}, b′::NTuple{N, Any}) where N
    if @generated
        return quote
            Base.@nexprs $N i -> a′_i = backward(lens.lenses[i], m[i], b′[i])
            return Base.@ntuple $N a′
        end

    else
        a′ = Vector(undef, N)
        for i in 1:N
            a′[i] = backward(lens.lenses[i], m[i], b′[i])
        end

        return Tuple(a′)
    end
end

inverse(lens::ProductLens{N}, b::NTuple{N, Any}) where N = inverse.(lens.lenses, b)

abstract type ParaLens end

function (lens::ParaLens)(a, p)
    _, b = forward(lens, a, p)
    return b
end

function (lens::ParaLens)(a, p, b′)
    m, b = forward(lens, a, p)
    a′, p′ = backward(lens, m, b′)
    return b, a′, p′
end

initialise(lens::ParaLens) = initialise(Random.default_rng(), lens)

struct LiftedParaLens{T <: Lens} <: ParaLens
    lens::T
end

lift(lens::Lens) = LiftedParaLens(lens)
lift(lens::ParaLens) = lens

forward(lens::LiftedParaLens, a, p) = forward(lens.lens, a)
backward(lens::LiftedParaLens, m, b′) = backward(lens.lens, m, b′), nothing
initialise(rng::AbstractRNG, lens::LiftedParaLens) = nothing

struct ComposedParaLens{N, T <: NTuple{N, ParaLens}} <: ParaLens
    lenses::T

    function ComposedParaLens(lenses...)
        flat = foldl(_composed_para, lenses; init=())
        return new{length(flat), typeof(flat)}(flat)
    end
end

_composed_para(xs::Tuple, x::ParaLens) = (xs..., x)
_composed_para(xs::Tuple, x::ComposedParaLens) = (xs..., x.lenses...)
_composed_para(xs::Tuple, x::Lens) = (xs..., lift(x))
_composed_para(xs::Tuple, x::ComposedLens) = (xs..., lift.(x.lenses)...)

function Base.:∘(outer::Union{Lens, ParaLens}, inner::Union{Lens, ParaLens})
    return ComposedParaLens(inner, outer)
end

function ⨟(inner::Union{Lens, ParaLens}, outer::Union{Lens, ParaLens})
    return ComposedParaLens(inner, outer)
end

function forward(lens::ComposedParaLens{N}, a, p::NTuple{N, Any}) where N
    if @generated
        return quote
            b_0 = a
            Base.@nexprs $N i -> (m_i, b_i) = forward(lens.lenses[i], b_{i - 1}, p[i])
            return (Base.@ntuple $N m), $(Symbol("b_", N))
        end

    else
        m = Vector(undef, N)
        for i in 1:N
            m[i], a = forward(lens.lenses[i], a, p[i])
        end

        return Tuple(m), a
    end
end

function backward(lens::ComposedParaLens{N}, m::NTuple{N, Any}, b′) where N
    if @generated
        return quote
            $(Symbol("a′_", N)) = b′
            @nexprs_reverse $N i -> (a′_{i - 1}, p′_i) = backward(lens.lenses[i], m[i], a′_i)
            return a′_0, (Base.@ntuple $N p′)
        end

    else
        p′ = Vector(undef, N)
        for i in N:-1:1
            b′, p′[i] = backward(lens.lenses[i], m[i], b′)
        end

        return b′, Tuple(p′)
    end
end

initialise(rng::AbstractRNG, lens::ComposedParaLens) = initialise.(rng, lens.lenses)

struct ProductParaLens{N, T <: NTuple{N, ParaLens}} <: ParaLens
    lenses::T

    function ProductParaLens(lenses...)
        flat = foldl(_product_para, lenses; init=())
        return new{length(flat), typeof(flat)}(flat)
    end
end

_product_para(xs::Tuple, x::ParaLens) = (xs..., x)
_product_para(xs::Tuple, x::ProductParaLens) = (xs..., x.lenses...)
_product_para(xs::Tuple, x::Lens) = (xs..., lift(x))
_product_para(xs::Tuple, x::ProductLens) = (xs..., lift.(x.lenses)...)

×(left::Union{Lens, ParaLens}, right::Union{Lens, ParaLens}) = ProductParaLens(left, right)

function forward(lens::ProductParaLens{N}, a::NTuple{N, Any}, p::NTuple{N, Any}) where N
    if @generated
        return quote
            Base.@nexprs $N i -> (m_i, b_i) = forward(lens.lenses[i], a[i], p[i])
            return (Base.@ntuple $N m), (Base.@ntuple $N b)
        end

    else
        m, b = Vector(undef, N), Vector(undef, N)
        for i in 1:N
            m[i], b[i] = forward(lens.lenses[i], a[i], p[i])
        end

        return Tuple(m), Tuple(b)
    end
end

function backward(lens::ProductParaLens{N}, m::NTuple{N, Any}, b′::NTuple{N, Any}) where N
    if @generated
        return quote
            Base.@nexprs $N i -> (a′_i, p′_i) = backward(lens.lenses[i], m[i], b′[i])
            return (Base.@ntuple $N a′), (Base.@ntuple $N p′)
        end

    else
        a′, p′ = Vector(undef, N), Vector(undef, N)
        for i in 1:N
            a′[i] = backward(lens.lenses[i], m[i], b′[i])
        end

        return Tuple(a′), Tuple(p′)
    end
end

initialise(rng::AbstractRNG, lens::ProductParaLens) = initialise.(rng, lens.lenses)

struct RemappedParaLens{S <: Lens, T <: ParaLens} <: ParaLens
    remap::S
    lens::T
end

RemappedParaLens(remap::Lens, lens::Lens) = RemappedParaLens(remap, lift(lens))
function RemappedParaLens(remap::Lens, lens::RemappedParaLens)
    return RemappedParaLens(remap ⨟ lens.remap, lens.lens)
end

remap(remap::Lens, lens) = RemappedParaLens(remap, lens)
(remap::Lens)(lens::Union{Lens, ParaLens}) = RemappedParaLens(remap, lens)

function forward(lens::RemappedParaLens, a, p)
    m₁, q = forward(lens.remap, p)
    m₂, b = forward(lens.lens, a, q)
    return (m₁, m₂), b
end

function backward(lens::RemappedParaLens, (m₁, m₂)::Tuple{Any, Any}, b′)
    a′, q′ = backward(lens.lens, m₂, b′)
    p′ = backward(lens.remap, m₁, q′)
    return a′, p′
end

function initialise(rng::AbstractRNG, lens::RemappedParaLens)
    return inverse(lens.remap, initialise(rng, lens.lens))
end
