# Weave Logprobs Reasoning Loop

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
- **Non-reasoning models with uncertainty loops** (e.g., GPT-4-mini with our framework)
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
- **Weave** for comprehensive experiment tracking and visualization
- **Perplexity-based thresholds** for triggering refinement
- **Top-k alternatives** for informing the model about uncertainty regions

### Core Components

```python
@weave.op()
def answer_difficult_question_with_uncertainty(
    question: str,
    model: str = "gpt-4-mini", 
    top_k: int = 5,
    threshold: float = 1.4,
    temperature: float = 0.2
):
    # Initial generation with logprobs
    # Perplexity calculation
    # Conditional refinement based on uncertainty
    # Returns structured metrics and final answer
```

## Getting Started

### Prerequisites
```bash
pip install -r requirements.txt
export OPENAI_API_KEY="your-api-key"
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

### Running the Python Script
```bash
python wb-logprobs.py
```

## Results & Insights

Our experiments show that:
1. **Uncertainty-aware loops can match reasoning model quality** at a fraction of the cost
2. **Perplexity effectively identifies problematic generations** before they reach users
3. **Top-k alternatives provide valuable context** for self-correction
4. **The transformer framework's exclusion of uncertainty metrics represents a significant limitation**

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

### The Transformer Framework Gap
The standard transformer inference pipeline:
- Discards logprobs after token selection
- Ignores uncertainty signals during generation
- Lacks self-correction mechanisms
- Provides no confidence metrics to downstream systems

Our approach addresses these limitations by treating uncertainty as a first-class citizen in the generation process.

## Technical Details

### Perplexity Calculation
```python
perplexity = exp(-mean(log_probabilities))
```

### Uncertainty Identification
- Sort tokens by log probability
- Extract top-k alternatives for most uncertain positions
- Generate structured uncertainty reports

### Refinement Strategy
- Provide original answer + uncertainty analysis
- Request targeted improvements for uncertain regions
- Lower temperature for refinement pass

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

**Project Status:** Active Development

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
