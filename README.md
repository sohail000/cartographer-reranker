# Cartographer : A Lightweight E-Commerce Reranker

A small, fast cross-encoder that scores how well a shopping query matches a product,
fine-tuned to beat the general-purpose reranker it replaces on real e-commerce search data.

**Model on Hugging Face:** [2013khansohail/cartographer-ecommerce-reranker-MiniLM-L6-v2](https://huggingface.co/2013khansohail/cartographer-ecommerce-reranker-MiniLM-L6-v2)


<img width="1774" height="887" alt="cartographer" src="https://github.com/user-attachments/assets/ff818b07-fa8c-455c-a46d-e67e54cffcf0" />


## What this is

Off-the-shelf rerankers like `cross-encoder/ms-marco-MiniLM-L-6-v2` were trained on web-search
passages, not product listings. They are decent but mediocre when the "passage" is a noisy
product title full of brands, attributes, and specs. Cartographer is the **same size and speed**
as that baseline (22.7M parameters), but fine-tuned on Amazon's public Shopping Queries Dataset
(ESCI) so it understands product relevance. It is meant as a drop-in upgrade, not a leaderboard
trophy.

The whole project in one honest sentence: the off-the-shelf model scored 0.7193 nDCG@10 on
shopping queries; this fine-tuned, same-size model scores 0.7642, and the repo has the numbers,
the ablations, and the reproduction steps.

## Result

Held-out ESCI test split (US / English): 8,956 queries, 181,701 (query, product) pairs.
Both models scored at identical settings.

| Metric    | Baseline | Cartographer | Δ relative |
|-----------|---------:|-------------:|-----------:|
| nDCG@5    | 0.6748   | **0.7272**   | +7.76%     |
| nDCG@10   | 0.7193   | **0.7642**   | +6.24%     |
| nDCG@20   | 0.8064   | **0.8354**   | +3.60%     |

The gain is largest at the top of the ranking, where it matters most for product search.
No regressions on any reported cutoff.

## Quick start

```python
from sentence_transformers import CrossEncoder

model = CrossEncoder("2013khansohail/cartographer-ecommerce-reranker-MiniLM-L6-v2", max_length=256)

query = "wireless noise cancelling headphones"
products = [
    "Sony WH-1000XM5 Wireless Noise Cancelling Headphones",
    "Wired Earbuds with Microphone, 3.5mm",
    "Headphone Stand Aluminum Desk Holder",
]
scores = model.predict([(query, p) for p in products])
for product, score in sorted(zip(products, scores), key=lambda x: x[1], reverse=True):
    print(f"{score:.3f}  {product}")
```

## How it was built

- **Base model:** `cross-encoder/ms-marco-MiniLM-L-6-v2`. Fine-tuning from an existing reranker
  checkpoint, rather than a raw backbone, converges faster and higher.
- **Data:** Amazon Shopping Queries Dataset (ESCI), US/English, `small_version` split. Product
  text is the product title (title-only recipe for v1).
- **Objective:** pointwise regression (MSE with sigmoid activation) onto graded relevance gains,
  mapped E=1.0, S=0.1, C=0.01, I=0.0.
- **Recipe:** 2 epochs with early stopping on validation nDCG@10, fp16, effective batch size 64,
  lr 2e-5. Trained on a single consumer GPU (NVIDIA RTX 3050, 4GB).

It learned the relevance gradation: on the test set it produces a clean E > S > C > I ordering
with the per-class score noise collapsed from std ~6 (baseline) to std ~1. Full per-label tables
and limitations are in the [model card](https://huggingface.co/2013khansohail/cartographer-ecommerce-reranker-MiniLM-L6-v2).

## Reproduce it

```bash
git clone https://github.com/sohail000/cartographer-reranker.git
cd cartographer-reranker
pip install -r requirements.txt
```

Then open the notebook and run the cells top to bottom. The notebook downloads ESCI, runs the
baseline ("before") number, fine-tunes the model, and runs the full evaluation ("after") against
the baseline. The ESCI data (~hundreds of MB) is downloaded by the notebook and is not committed
to this repo; neither are the model weights, which live on Hugging Face.

## Honest limitations

- ESCI is Amazon data, so the model inherits Amazon's product distribution and may need further
  tuning for other catalogs.
- Substitute vs. Complement is the model's softest boundary; their score distributions overlap.
- This is the best model in its weight class, not the best model available. Some hard queries
  will need a heavier reranker.
- English-only, title-only product text in this version.

## License

MIT. Built on the Amazon Shopping Queries Dataset (ESCI),
[amazon-science/esci-data](https://github.com/amazon-science/esci-data).
