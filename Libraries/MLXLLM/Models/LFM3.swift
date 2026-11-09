//
//  LFM3.swift
//  mlx-swift-lm
//
//  Backport of upstream ml-explore/mlx-swift-lm c48e55e (#750, "Add LiquidAI LFM3 model
//  support") onto the 0.30.6-based fork. Upstream needs mlx-swift >= 0.33, which needs
//  Swift tools 6.3; this port uses only APIs present in 0.30.6.
//
//  Port of https://huggingface.co/LiquidAI/LFM3-1.7B/blob/main/modeling_lfm3.py
//
//  LFM3 is LFM2 with sliding-window attention as a third layer type, a sigmoid gate per
//  attention head, partial RoPE on full-attention layers, and an optional RMSNorm on each
//  block output ("sandwich norm").
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

public struct LFM3Configuration: Codable, Sendable {
    public enum LayerType: String, Codable, Sendable {
        case conv
        case fullAttention = "full_attention"
        case slidingAttention = "sliding_attention"
    }

    struct RopeParameters: Codable, Sendable {
        let ropeType: String
        let ropeTheta: Float
        let partialRotaryFactor: Float

        enum CodingKeys: String, CodingKey {
            case ropeType = "rope_type"
            case ropeTheta = "rope_theta"
            case partialRotaryFactor = "partial_rotary_factor"
        }

        init(ropeTheta: Float, partialRotaryFactor: Float = 1.0) {
            self.ropeType = "default"
            self.ropeTheta = ropeTheta
            self.partialRotaryFactor = partialRotaryFactor
        }

        init(from decoder: Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            self.ropeType =
                try container.decodeIfPresent(String.self, forKey: .ropeType) ?? "default"
            self.ropeTheta = try container.decode(Float.self, forKey: .ropeTheta)
            self.partialRotaryFactor =
                try container.decodeIfPresent(Float.self, forKey: .partialRotaryFactor) ?? 1.0
        }
    }

    let modelType: String
    public let vocabularySize: Int
    public let hiddenSize: Int
    public let hiddenLayers: Int
    public let attentionHeads: Int
    public let kvHeads: Int
    public let headDim: Int
    public let intermediateSize: Int
    let maxPositionEmbeddings: Int?
    let normEps: Float
    let convBias: Bool
    let convLCache: Int
    public let layerTypes: [LayerType]
    public let slidingWindow: Int
    let useSandwichNorm: Bool
    public let tieWordEmbeddings: Bool
    let ropeParameters: [String: RopeParameters]

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabularySize = "vocab_size"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case intermediateSize = "intermediate_size"
        case maxPositionEmbeddings = "max_position_embeddings"
        case normEps = "norm_eps"
        case convBias = "conv_bias"
        case convLCache = "conv_L_cache"
        case layerTypes = "layer_types"
        case slidingWindow = "sliding_window"
        case useSandwichNorm = "use_sandwich_norm"
        case tieWordEmbeddings = "tie_word_embeddings"
        case ropeParameters = "rope_parameters"
    }

    /// The reference reads the LFM2 spelling `tie_embedding` before `tie_word_embeddings`.
    private enum AltCodingKeys: String, CodingKey {
        case tieEmbedding = "tie_embedding"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        self.modelType = try container.decodeIfPresent(String.self, forKey: .modelType) ?? "lfm3"
        self.vocabularySize =
            try container.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 128_000
        self.hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        self.hiddenLayers = try container.decode(Int.self, forKey: .hiddenLayers)
        self.attentionHeads = try container.decode(Int.self, forKey: .attentionHeads)
        self.kvHeads = try container.decode(Int.self, forKey: .kvHeads)
        self.headDim =
            try container.decodeIfPresent(Int.self, forKey: .headDim)
            ?? hiddenSize / attentionHeads
        self.intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)
        self.maxPositionEmbeddings = try container.decodeIfPresent(
            Int.self, forKey: .maxPositionEmbeddings)
        self.normEps = try container.decodeIfPresent(Float.self, forKey: .normEps) ?? 1e-5
        self.convBias = try container.decodeIfPresent(Bool.self, forKey: .convBias) ?? false
        self.convLCache = try container.decodeIfPresent(Int.self, forKey: .convLCache) ?? 7
        self.slidingWindow =
            try container.decodeIfPresent(Int.self, forKey: .slidingWindow) ?? 256
        self.useSandwichNorm =
            try container.decodeIfPresent(Bool.self, forKey: .useSandwichNorm) ?? true
        let alt = try decoder.container(keyedBy: AltCodingKeys.self)
        self.tieWordEmbeddings =
            try alt.decodeIfPresent(Bool.self, forKey: .tieEmbedding)
            ?? container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? true

        // Same default pattern as the reference: full attention every fourth layer from
        // the third, sliding attention in between, short convolutions elsewhere.
        let hiddenLayers = self.hiddenLayers
        self.layerTypes =
            try container.decodeIfPresent([LayerType].self, forKey: .layerTypes)
            ?? (0 ..< hiddenLayers).map { i in
                i % 4 == 2 ? .fullAttention : (i % 4 == 0 && i > 0 ? .slidingAttention : .conv)
            }
        guard layerTypes.count == hiddenLayers else {
            throw DecodingError.dataCorruptedError(
                forKey: .layerTypes, in: container,
                debugDescription:
                    "layer_types has \(layerTypes.count) entries but num_hidden_layers is \(hiddenLayers)"
            )
        }

        self.ropeParameters =
            try container.decodeIfPresent([String: RopeParameters].self, forKey: .ropeParameters)
            ?? [
                LayerType.fullAttention.rawValue: .init(
                    ropeTheta: 1_000_000, partialRotaryFactor: 0.5),
                LayerType.slidingAttention.rawValue: .init(ropeTheta: 10_000),
            ]
        for layerType in Set(layerTypes) where layerType != .conv {
            guard let rope = ropeParameters[layerType.rawValue] else {
                throw DecodingError.dataCorruptedError(
                    forKey: .ropeParameters, in: container,
                    debugDescription: "rope_parameters has no entry for \(layerType.rawValue)")
            }
            // Scaled RoPE types are not ported; running them as default would be wrong.
            guard rope.ropeType == "default" else {
                throw DecodingError.dataCorruptedError(
                    forKey: .ropeParameters, in: container,
                    debugDescription: "Unsupported LFM3 rope_type '\(rope.ropeType)'")
            }
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(modelType, forKey: .modelType)
        try container.encode(vocabularySize, forKey: .vocabularySize)
        try container.encode(hiddenSize, forKey: .hiddenSize)
        try container.encode(hiddenLayers, forKey: .hiddenLayers)
        try container.encode(attentionHeads, forKey: .attentionHeads)
        try container.encode(kvHeads, forKey: .kvHeads)
        try container.encode(headDim, forKey: .headDim)
        try container.encode(intermediateSize, forKey: .intermediateSize)
        try container.encodeIfPresent(maxPositionEmbeddings, forKey: .maxPositionEmbeddings)
        try container.encode(normEps, forKey: .normEps)
        try container.encode(convBias, forKey: .convBias)
        try container.encode(convLCache, forKey: .convLCache)
        try container.encode(layerTypes, forKey: .layerTypes)
        try container.encode(slidingWindow, forKey: .slidingWindow)
        try container.encode(useSandwichNorm, forKey: .useSandwichNorm)
        try container.encode(tieWordEmbeddings, forKey: .tieWordEmbeddings)
        try container.encode(ropeParameters, forKey: .ropeParameters)
    }
}

/// Partial RoPE as LFM3 defines it: the first `partialRotaryFactor` of EACH HALF of the head
/// is rotated, with the frequencies of the full head. `RoPE(dimensions:)` would instead rotate
/// a contiguous prefix with frequencies for the shorter width — a different function.
/// Port of upstream `ProportionalRoPE` (RoPEUtils.swift), which 0.30.6 does not have.
final class LFM3ProportionalRoPE: Module {
    let dims: Int
    let rotatedDims: Int
    let base: Float
    let freqs: MLXArray?

    init(dims: Int, base: Float, partialRotaryFactor: Float) {
        self.dims = dims
        self.base = base
        let ropeAngles = Int(partialRotaryFactor * Float(dims) / 2.0)
        self.rotatedDims = 2 * ropeAngles
        if rotatedDims > 0 {
            let exponents =
                MLXArray(stride(from: 0, to: rotatedDims, by: 2)).asType(.float32) / Float(dims)
            self.freqs = MLX.pow(base, exponents)
        } else {
            self.freqs = nil
        }
        super.init()
    }

    func callAsFunction(_ x: MLXArray, offset: Int = 0) -> MLXArray {
        guard rotatedDims > 0, let freqs else { return x }

        let half = dims / 2
        let rotatedHalf = rotatedDims / 2

        let headParts = split(x, indices: [half], axis: -1)
        let leftParts = split(headParts[0], indices: [rotatedHalf], axis: -1)
        let rightParts = split(headParts[1], indices: [rotatedHalf], axis: -1)

        var rotated = concatenated([leftParts[0], rightParts[0]], axis: -1)
        rotated = MLXFast.RoPE(
            rotated, dimensions: rotatedDims, traditional: false, base: nil, scale: 1.0,
            offset: offset, freqs: freqs)
        let rotatedParts = split(rotated, indices: [rotatedHalf], axis: -1)

        let left = concatenated([rotatedParts[0], leftParts[1]], axis: -1)
        let right = concatenated([rotatedParts[1], rightParts[1]], axis: -1)
        return concatenated([left, right], axis: -1)
    }
}

/// The two RoPE shapes LFM3 uses, behind one call so the attention block stays simple.
enum LFM3Rope {
    case full(RoPE)
    case proportional(LFM3ProportionalRoPE)

    func callAsFunction(_ x: MLXArray, offset: Int) -> MLXArray {
        switch self {
        case .full(let rope): rope(x, offset: offset)
        case .proportional(let rope): rope(x, offset: offset)
        }
    }
}

final class LFM3Attention: Module {
    let attentionHeads: Int
    let kvHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "gate_proj") var gateProj: Linear

    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    let rope: LFM3Rope

    init(_ args: LFM3Configuration, layerType: LFM3Configuration.LayerType) {
        self.attentionHeads = args.attentionHeads
        self.kvHeads = args.kvHeads
        self.headDim = args.headDim
        self.scale = pow(Float(headDim), -0.5)

        let dim = args.hiddenSize
        _qProj.wrappedValue = Linear(dim, attentionHeads * headDim, bias: false)
        _kProj.wrappedValue = Linear(dim, kvHeads * headDim, bias: false)
        _vProj.wrappedValue = Linear(dim, kvHeads * headDim, bias: false)
        _oProj.wrappedValue = Linear(attentionHeads * headDim, dim, bias: false)
        _gateProj.wrappedValue = Linear(dim, attentionHeads, bias: false)

        _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.normEps)
        _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.normEps)

        // Presence is validated in LFM3Configuration.init for every attention layer type.
        let ropeParameters = args.ropeParameters[layerType.rawValue]
            ?? .init(ropeTheta: 10_000)
        if ropeParameters.partialRotaryFactor < 1 {
            self.rope = .proportional(
                LFM3ProportionalRoPE(
                    dims: headDim, base: ropeParameters.ropeTheta,
                    partialRotaryFactor: ropeParameters.partialRotaryFactor))
        } else {
            self.rope = .full(
                RoPE(dimensions: headDim, traditional: false, base: ropeParameters.ropeTheta))
        }
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        var queries = qNorm(qProj(x).reshaped(B, L, attentionHeads, -1)).transposed(0, 2, 1, 3)
        var keys = kNorm(kProj(x).reshaped(B, L, kvHeads, -1)).transposed(0, 2, 1, 3)
        let values = vProj(x).reshaped(B, L, kvHeads, -1).transposed(0, 2, 1, 3)

        let offset = cache?.offset ?? 0
        queries = rope(queries, offset: offset)
        keys = rope(keys, offset: offset)

        let output = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: cache,
            scale: scale,
            mask: mask
        )
        .transposed(0, 2, 1, 3)

        // One sigmoid gate per head, computed from the block input.
        let gate = sigmoid(gateProj(x)).expandedDimensions(axis: -1)
        return oProj((output * gate).reshaped(B, L, -1))
    }
}

final class LFM3ShortConv: Module {
    let hiddenSize: Int
    let lCache: Int

    @ModuleInfo(key: "conv") var conv: Conv1d
    @ModuleInfo(key: "in_proj") var inProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear

    init(_ args: LFM3Configuration) {
        self.hiddenSize = args.hiddenSize
        self.lCache = args.convLCache

        _conv.wrappedValue = Conv1d(
            inputChannels: hiddenSize,
            outputChannels: hiddenSize,
            kernelSize: lCache,
            groups: hiddenSize,
            bias: args.convBias
        )
        _inProj.wrappedValue = Linear(hiddenSize, 3 * hiddenSize, bias: args.convBias)
        _outProj.wrappedValue = Linear(hiddenSize, hiddenSize, bias: args.convBias)
    }

    func callAsFunction(_ x: MLXArray, cache: MambaCache?) -> MLXArray {
        let BCx = inProj(x).split(parts: 3, axis: -1)
        let (B, C) = (BCx[0], BCx[1])
        var Bx = B * BCx[2]

        // The cache holds the last `lCache - 1` inputs, so the conv needs no padding.
        let state =
            cache?[0] ?? MLXArray.zeros([Bx.dim(0), lCache - 1, hiddenSize], dtype: Bx.dtype)
        Bx = concatenated([state, Bx], axis: -2)
        if let cache {
            cache[0] = contiguous(Bx[0..., (Bx.dim(1) - (lCache - 1))..., 0...])
        }

        return outProj(C * conv(Bx))
    }
}

final class LFM3MLP: Module, UnaryLayer {
    @ModuleInfo(key: "w1") var w1: Linear
    @ModuleInfo(key: "w2") var w2: Linear
    @ModuleInfo(key: "w3") var w3: Linear

    init(dim: Int, hiddenDim: Int) {
        _w1.wrappedValue = Linear(dim, hiddenDim, bias: false)
        _w2.wrappedValue = Linear(hiddenDim, dim, bias: false)
        _w3.wrappedValue = Linear(dim, hiddenDim, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        w2(silu(w1(x)) * w3(x))
    }
}

final class LFM3DecoderLayer: Module {
    let layerType: LFM3Configuration.LayerType

    @ModuleInfo(key: "self_attn") var attention: LFM3Attention?
    @ModuleInfo(key: "conv") var conv: LFM3ShortConv?
    @ModuleInfo(key: "feed_forward") var feedForward: LFM3MLP
    @ModuleInfo(key: "operator_norm") var operatorNorm: RMSNorm
    @ModuleInfo(key: "ffn_norm") var ffnNorm: RMSNorm
    @ModuleInfo(key: "post_operator_norm") var postOperatorNorm: RMSNorm?
    @ModuleInfo(key: "post_ffn_norm") var postFFNNorm: RMSNorm?

    init(_ args: LFM3Configuration, layerIdx: Int) {
        self.layerType = args.layerTypes[layerIdx]

        if layerType == .conv {
            _conv.wrappedValue = LFM3ShortConv(args)
        } else {
            _attention.wrappedValue = LFM3Attention(args, layerType: layerType)
        }

        _feedForward.wrappedValue = LFM3MLP(dim: args.hiddenSize, hiddenDim: args.intermediateSize)
        _operatorNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.normEps)
        _ffnNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.normEps)
        if args.useSandwichNorm {
            _postOperatorNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.normEps)
            _postFFNNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.normEps)
        }
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        var r: MLXArray
        if let attention {
            r = attention(operatorNorm(x), mask: mask, cache: cache)
        } else if let conv {
            r = conv(operatorNorm(x), cache: cache as? MambaCache)
        } else {
            // Unreachable: init builds exactly one of the two operators for every layer.
            r = MLXArray.zeros(like: x)
        }
        if let postOperatorNorm {
            r = postOperatorNorm(r)
        }
        let h = x + r

        var f = feedForward(ffnNorm(h))
        if let postFFNNorm {
            f = postFFNNorm(f)
        }
        return h + f
    }
}

/// The LFM3 text backbone: token embedding, decoder layers, and the final norm.
public final class LFM3ModelInner: Module {
    let args: LFM3Configuration
    let firstFullAttentionIdx: Int?
    let firstSlidingAttentionIdx: Int?

    let layers: [LFM3DecoderLayer]

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "embedding_norm") var embeddingNorm: RMSNorm

    init(_ args: LFM3Configuration) {
        self.args = args
        self.firstFullAttentionIdx = args.layerTypes.firstIndex(of: .fullAttention)
        self.firstSlidingAttentionIdx = args.layerTypes.firstIndex(of: .slidingAttention)

        _embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)

        self.layers = (0 ..< args.hiddenLayers).map { i in
            LFM3DecoderLayer(args, layerIdx: i)
        }

        _embeddingNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.normEps)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var h = embedTokens(inputs)
        let n = h.dim(1)

        let fullMask = firstFullAttentionIdx.map { makeAttentionMask(n: n, cache: cache?[$0]) }
        let slidingMask = firstSlidingAttentionIdx.map {
            makeAttentionMask(n: n, cache: cache?[$0], windowSize: args.slidingWindow)
        }

        for (i, layer) in layers.enumerated() {
            let mask: MLXFast.ScaledDotProductAttentionMaskMode =
                switch layer.layerType {
                case .fullAttention: fullMask ?? .causal
                case .slidingAttention: slidingMask ?? .causal
                case .conv: .none
                }
            h = layer(h, mask: mask, cache: cache?[i])
        }

        return embeddingNorm(h)
    }
}

public final class LFM3Model: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    public let model: LFM3ModelInner
    let configuration: LFM3Configuration

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ args: LFM3Configuration) {
        self.configuration = args
        self.vocabularySize = args.vocabularySize
        self.kvHeads = args.layerTypes.map { $0 == .conv ? 0 : args.kvHeads }
        self.model = LFM3ModelInner(args)
        if !args.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabularySize, bias: false)
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let hidden = model(inputs, cache: cache)
        return lmHead?(hidden) ?? model.embedTokens.asLinear(hidden)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized: [String: MLXArray] = [:]
        for (key, value) in weights {
            // MLX conversions nest the text model under `language_model`. Keep only the text
            // model, which drops any vision or audio tensors.
            var name = key
            if name.hasPrefix("language_model.") {
                name.removeFirst("language_model.".count)
            }
            let root = name.prefix { $0 != "." }
            guard root == "model" || root == "lm_head" else { continue }
            // A tied checkpoint can still carry an lm_head copy; the module has none to load.
            if configuration.tieWordEmbeddings, root == "lm_head" { continue }

            var param = value
            // PyTorch stores the depthwise conv as [C, 1, K]; MLX expects [C, K, 1].
            if name.hasSuffix("conv.conv.weight"), param.dim(-1) > param.dim(1) {
                param = param.transposed(0, 2, 1)
            }
            sanitized[name] = param
        }
        return sanitized
    }

    /// One cache per layer: a conv state for conv layers, a rotating cache bounded by the
    /// sliding window for sliding attention, and a full (or `maxKVSize`-bounded) cache for
    /// full attention.
    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        configuration.layerTypes.map { layerType in
            switch layerType {
            case .conv:
                MambaCache()
            case .slidingAttention:
                RotatingKVCache(maxSize: configuration.slidingWindow, keep: 0)
            case .fullAttention:
                if let maxKVSize = parameters?.maxKVSize {
                    RotatingKVCache(maxSize: maxKVSize, keep: 4)
                } else {
                    KVCacheSimple()
                }
            }
        }
    }
}

extension LFM3Model: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}
