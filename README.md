# Logprobs Reasoning Loop with Weights & Biases Weave, an observability tool 

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/monostate/weave-logprobs-reasoning-loop/blob/main/wb-logprobs.ipynb)

## Uncertainty-Aware Generation with OpenAI's Responses API

This project demonstrates a novel approach to improving AI model reasoning by leveraging token-level uncertainty metrics (logprobs) to create self-correcting generation loops. We compare this uncertainty-aware approach against traditional reasoning models to test whether explicit uncertainty handling can match or exceed the performance of dedicated reasoning architectures.

## Core Concept

Modern transformers typically discard valuable uncertainty information during inference. This project explores whether we can harness this discarded information—specifically logprobs and top-k alternatives—to create more reliable and accurate AI responses without requiring specialized reasoning models.

### Key Innovation

We implement an **uncertainty-aware generation loop** that:
1. Generates an initial response while tracking token-level uncertainty (perplexity)
2. Automatically identifies regions of high uncertainty using logprobs
3. Triggers a refinement pass when uncertainty exceeds a threshold
4. Provides the model with explicit information about uncertain tokens and their alternatives
5. Produces a refined, more accurate final response

## What We're Testing

### Hypothesis
**Uncertainty metrics (logprobs) and top-k alternatives contain valuable reasoning signals that current transformer frameworks underutilize.**

### Comparison
- **Non-reasoning models with uncertainty loops** (e.g., gpt-4.1-mini with our framework)
- **Native reasoning models** (e.g., o4-mini) - Note: These don't expose logprobs, so uncertainty analysis is not available

### Metrics Tracked
- Token-level perplexity
- Average log probabilities
- Response accuracy
- Token usage and costs
- Generation time

## Technical Implementation

The project uses:
- **OpenAI Responses API** with `include=["message.output_text.logprobs"]`
- **[Weave by Weights & Biases](https://wandb.ai/site/weave)** for comprehensive experiment tracking and visualization
- **Perplexity-based thresholds** for triggering refinement
- **Top-k alternatives** for informing the model about uncertainty regions

### Why Weave?

Weave is essential for this project because it provides:
- **Persistent experiment tracking** - Every run, metric, and decision is logged and queryable
- **Hierarchical operation tracing** - See exactly how the uncertainty loop makes decisions
- **Production-ready observability** - Transform research experiments into deployable products
- **Free tier available** - Get started without any cost commitment

**Get your free Weave API key at: [https://wandb.ai/authorize](https://wandb.ai/authorize)**

Weave enables us to:
1. Track every token's uncertainty metrics across experiments
2. Compare refinement decisions and their impacts
3. Build a dataset of uncertainty patterns for future research
4. Create reproducible experiments with full lineage tracking
5. Visualize the relationship between uncertainty and answer quality

### Core Components

```python
@weave.op()
def answer_difficult_question_with_uncertainty(
    question: str,
    model: str = "gpt-4.1-mini", 
    top_k: int = 5,
    threshold: float = 1.4,
    temperature: float = 0.2
):
    # Initial generation with logprobs
    # Calculate multiple uncertainty metrics:
    #   - Perplexity from average logprobs
    #   - Maximum entropy across tokens
    #   - Count of low-confidence tokens
    # Multi-metric refinement trigger
    # Conditional refinement with detailed uncertainty report
    # Returns structured metrics and final answer
```

### Enhanced Uncertainty Detection

Our implementation now uses multiple complementary metrics:

1. **Perplexity**: `exp(-mean(log_probabilities))` - Overall uncertainty measure
2. **Token-level Entropy**: `-sum(p * log(p))` across top-k alternatives
3. **Confidence Distribution**: Count of tokens below confidence thresholds
4. **Contextual Analysis**: Shows uncertain tokens with surrounding context

## Getting Started

### Prerequisites

This project includes a vendorized version of `polyfile-weave` with fixes for Python 3.9+ compatibility.

#### Setting up Virtual Environment (Required)
```bash
# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install dependencies (includes local polyfile-weave)
pip install -r requirements.txt

# Set up environment variables
cp env.example .env
# Edit .env with your API keys
```

#### Setting up Weave Tracking (Recommended)

Weave provides essential observability for understanding how the uncertainty loop works:

1. **Get your free API key**: Visit [https://wandb.ai/authorize](https://wandb.ai/authorize)
2. **Add to your .env file**:
   ```bash
   WANDB_API_KEY=your-api-key-here
   WEAVE_PROJECT=weave-intro-notebook  # or your custom project name
   ```
3. **View your experiments**: After running, visit the URL printed in console to explore:
   - Token-by-token uncertainty metrics
   - Refinement decision rationale
   - Cost and performance comparisons
   - Full conversation traces with hierarchical operations

The free tier includes:
- Unlimited public projects
- 100GB of storage
- Full access to Weave features
- No credit card required

**Note:** 
- The vendorized `polyfile-weave` package is included to fix compatibility issues with reserved keywords in the upstream package.
- The script includes a runtime patch for Weave to enable gql 4.0+ compatibility (see [our PR](https://github.com/wandb/weave/pull/new/fix/gql-4-compatibility-and-dependency-separation) for the permanent fix).

### Running Locally (Python Script)
```bash
# Option 1: Use .env file (recommended)
# Edit .env with your OPENAI_API_KEY
python wb-logprobs.py

# Option 2: Export environment variable
export OPENAI_API_KEY="sk-your-key-here"
python wb-logprobs.py

# Option 3: Pass a custom question
python wb-logprobs.py "Explain the halting problem and its implications"
```

### Troubleshooting

**Weave Initialization Error:**
If you encounter a `TypeError` when initializing Weave:
```bash
# Option 1: Install compatible gql version
pip install gql==3.4.1

# Option 2: Simply run the notebook - it will automatically handle the error
# The notebook includes fallback handling and can run without W&B tracking
```

**Reasoning Model Compatibility:**
The code automatically handles differences between reasoning models (o1, o4) and standard models:
- Reasoning models don't support `temperature` or `logprobs` parameters
- The code detects model type and adjusts API calls accordingly
- Reasoning models won't have uncertainty metrics or refinement loops (no logprobs available)
- Both model types will run successfully for comparison purposes

The notebook is designed to run even if Weave initialization fails, so you can proceed with the uncertainty experiments regardless of tracking setup.

### Running the Notebook
```bash
jupyter notebook wb-logprobs.ipynb
```

## Results & Insights

### Performance Benchmarks

Our comprehensive testing reveals impressive results:

#### Cost Efficiency
- **gpt-4.1-mini with uncertainty loop**: 30-43% of o4-mini reasoning model cost
- Average cost per complex question: $0.0007-$0.0011 vs $0.0019-$0.0058

#### Quality Metrics
Testing on controversial and complex questions (AGI predictions, ethical implications, cryptocurrency debates):
- **Comparable answer quality** to reasoning models
- **Improved confidence calibration** through explicit uncertainty handling
- **Reduced hallucination** via targeted refinement

#### Refinement Triggers
Our multi-metric approach catches uncertainty that single metrics miss:
- Perplexity threshold (>1.4)
- Maximum entropy (>1.5) 
- High uncertainty token count (≥3 tokens <50% confidence)

#### API Performance Analysis
Discovered significant performance characteristics:
- Simple questions: 2-6 seconds (faster than reasoning models)
- Complex technical questions: 54-67 seconds (API limitation, not our code)
- The more powerful the model, the slower the response (gpt-4.1: 99s, gpt-4o: 61s, gpt-4.1-mini: 67s)

### Key Findings

1. **2.75x cost reduction** compared to reasoning models while maintaining quality
2. **Intelligent refinement** - only triggers when genuinely uncertain (not for all responses)
3. **Rich uncertainty analysis** provides context about specific uncertain tokens and alternatives
4. **Hierarchical logging** via Weave enables deep analysis of the decision process

## Future Roadmap

### Phase 1: Extended Uncertainty Metrics
- Integrate pre-softmax hidden states
- Incorporate raw logits analysis
- Develop multi-layer uncertainty aggregation

### Phase 2: Full Inference Framework
- Build a production-ready inference server
- Implement streaming with real-time uncertainty monitoring
- Create adaptive thresholds based on task complexity

### Phase 3: Model-Agnostic Implementation
- Extend beyond OpenAI to open-source models
- Support for local inference with uncertainty extraction
- Develop uncertainty-aware fine-tuning methods

### Phase 4: Advanced Applications
- Multi-turn conversation uncertainty tracking
- Uncertainty-guided retrieval augmentation
- Collaborative uncertainty resolution across model ensembles

## Key Insights

### Why This Matters
Current transformer architectures make discrete token selections, discarding the rich probability distributions that could inform better reasoning. By capturing and utilizing this uncertainty information, we can:

1. **Reduce hallucinations** by identifying when models are uncertain
2. **Improve accuracy** through targeted refinement
3. **Lower costs** compared to dedicated reasoning models
4. **Provide transparency** about model confidence

### The Power of Observable AI with Weave

This project demonstrates how Weave transforms experimental AI research into production-ready systems:

**For Researchers:**
- Every experiment is automatically versioned and comparable
- Uncertainty patterns become queryable datasets
- Collaborate with full experiment reproducibility
- Build on previous results without losing context

**For Product Builders:**
- Monitor uncertainty metrics in production
- Set alerts for high-uncertainty responses
- A/B test different uncertainty thresholds
- Track cost-performance tradeoffs in real-time

**Data Persistence Benefits:**
- All logprobs and uncertainty metrics are stored permanently
- Build training datasets from real uncertainty patterns
- Analyze long-term trends in model confidence
- Create uncertainty benchmarks for new models

### The Transformer Framework Gap
The standard transformer inference pipeline:
- Discards logprobs after token selection
- Ignores uncertainty signals during generation
- Lacks self-correction mechanisms
- Provides no confidence metrics to downstream systems

Our approach addresses these limitations by treating uncertainty as a first-class citizen in the generation process.

## Technical Details

For a comprehensive technical deep-dive including:
- Mathematical formulas and derivations
- Complete implementation details
- API response processing
- Example uncertainty reports
- Performance analysis

**See [TECHNICAL.md](TECHNICAL.md)**

### Quick Overview

**Perplexity**: `exp(-mean(log_probabilities))` - Overall uncertainty measure

**Entropy**: `-sum(p * log(p))` - Token-level uncertainty quantification

**Decision Logic**: Refinement triggers if:
- Perplexity > 1.4 OR
- Max entropy > 1.5 OR  
- 3+ tokens with <50% confidence

**Observability**: Hierarchical `@weave.op()` tracking captures every decision and metric

## Contributing

We welcome contributions! Areas of particular interest:
- Alternative uncertainty metrics
- Multi-model uncertainty aggregation
- Visualization improvements
- Benchmark datasets for uncertainty-aware generation

## References

- [OpenAI Responses API Documentation](https://platform.openai.com/docs/api-reference/responses)
- [Weave: LLM Application Development Framework](https://weave-docs.wandb.ai/)
- [Information Theory and Neural Networks](https://www.inference.org.uk/mackay/itila/)

## License

MIT License - See LICENSE file for details

## Acknowledgments

- OpenAI for providing logprobs access via their APIs
- Weights & Biases team for the Weave framework
- The broader AI research community exploring uncertainty quantification

---

**Project Status:** Active Development (Phase 1: Benchmark Validation in Progress - August 2025)

**Contact:** andrew@monostate.ai or open an issue for questions or collaboration opportunities

**Citation:** If you use this work in your research, please cite:
```bibtex
@software{weave_logprobs_reasoning,
  title = {Uncertainty-Aware Generation with Logprobs},
  author = {Monostate},
  year = {2025},
  url = {https://github.com/monostate/weave-logprobs-reasoning-loop}
}
```

## Roadmap: Next Steps & Validation

### Immediate Next Steps (August 2025)
We are currently working on:
1. **Running ARC-AGI benchmarks** to validate abstract reasoning capabilities
2. **Testing on LogiQA 2.0** for logical reasoning validation
3. **GSM8K evaluation** to compare math problem-solving with o4-mini
4. **Setting up automated benchmark pipeline** with Weave tracking

### Phase 1: Benchmark Validation (Q3 2025 - Current)

#### Reasoning Benchmarks
- **[ARC-AGI](https://github.com/fchollet/ARC-AGI)** - Abstract reasoning corpus
- **[LogiQA 2.0](https://github.com/csitfun/LogiQA2.0)** - Logical reasoning in natural language
- **[GSM8K](https://github.com/openai/grade-school-math)** - Grade school math word problems
- **[MATH](https://github.com/hendrycks/math)** - Competition mathematics
- **[BigBench Hard](https://github.com/suzgunmirac/BIG-Bench-Hard)** - Challenging tasks from BIG-Bench
- **[MMLU](https://github.com/hendrycks/test)** - Massive multitask language understanding
- **[HumanEval](https://github.com/openai/human-eval)** - Code generation benchmarks

**Goal**: Demonstrate that uncertainty-aware loops achieve comparable or superior performance to reasoning models at 30-40% of the cost.

### Phase 2: Agentic Applications (Q4 2025)

#### Browser Automation Tasks
- **[WebArena](https://webarena.dev/)** - Realistic web navigation tasks
- **[Mind2Web](https://osu-nlp-group.github.io/Mind2Web/)** - Web interaction benchmarks
- Custom browser automation with uncertainty-driven exploration

#### Tool Use & Function Calling
- API integration with uncertainty-aware retries
- Database query generation with confidence metrics
- File system operations with safety checks based on uncertainty

#### Multi-Step Planning
- Task decomposition with uncertainty propagation
- Hierarchical planning with confidence thresholds
- Rollback mechanisms triggered by high uncertainty

### Phase 3: Chain-of-Thought Enhancement (Q4 2025 - Q1 2026)

#### Explicit Reasoning Traces
- **Uncertainty-guided CoT**: Use logprobs to identify where reasoning needs expansion
- **Selective verbalization**: Only elaborate on uncertain reasoning steps
- **Confidence-weighted chains**: Weight reasoning paths by aggregate certainty

#### Comparison Studies
- Standard CoT vs Uncertainty-aware CoT
- Few-shot prompting with uncertainty examples
- Zero-shot reasoning with automatic uncertainty detection

### Phase 4: Advanced Techniques (Q1 2026)

#### Self-Consistency with Uncertainty
- Multiple sampling with uncertainty aggregation
- Weighted voting based on path confidence
- Early stopping when uncertainty converges

#### Uncertainty-Aware Ensembles
- Multi-model uncertainty aggregation
- Cross-model confidence calibration
- Selective model routing based on uncertainty profiles

#### Active Learning Integration
- Identify high-uncertainty examples for human annotation
- Build uncertainty-aware training datasets
- Fine-tune models on uncertainty patterns

### Phase 5: Production Systems (Q1-Q2 2026)

#### Real-World Deployments
- **Customer Support**: Route uncertain queries to human agents
- **Content Generation**: Flag potentially problematic content based on uncertainty
- **Medical/Legal AI**: Mandatory uncertainty disclosure for high-stakes decisions
- **Educational Tools**: Adapt explanations based on model confidence

#### Infrastructure Development
- Streaming uncertainty detection
- Real-time refinement triggers
- Uncertainty-aware caching strategies
- Cost optimization with dynamic thresholds

### Phase 6: Research Extensions (Q2 2026 - Ongoing)

#### Theoretical Analysis
- Information-theoretic bounds on uncertainty reduction
- Optimal threshold learning algorithms
- Uncertainty propagation in multi-turn conversations

#### Novel Architectures
- Uncertainty-aware transformer variants
- Built-in refinement mechanisms
- Native uncertainty quantification layers

#### Cross-Domain Transfer
- Uncertainty patterns across different domains
- Domain-specific threshold calibration
- Transfer learning for uncertainty detection

## Validation Metrics

### Performance Targets
- **Accuracy**: Match or exceed reasoning model baselines
- **Cost**: Maintain 30-40% cost ratio vs reasoning models
- **Latency**: Optimize for <2x latency of single-pass generation
- **Reliability**: <5% false positive refinement rate

### Success Criteria
1. **Benchmark Performance**: Within 5% of reasoning model scores
2. **Cost Efficiency**: Consistent 2.5-3x cost reduction
3. **User Studies**: Preference for uncertainty-aware responses in blind tests
4. **Production Metrics**: Reduced error rates in deployed systems

## Community Collaboration

We invite researchers and practitioners to:
- **Contribute benchmark results** with your models and domains
- **Share uncertainty patterns** discovered in your applications
- **Propose new metrics** for uncertainty quantification
- **Build integrations** with other frameworks and tools

Join our efforts to make AI systems more reliable through uncertainty awareness!
