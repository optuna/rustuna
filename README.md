# Rustuna

:link: [**Website**](https://optuna.org/)
| :page_with_curl: [**Docs**](https://rustuna.readthedocs.io/)
| [**Twitter**](https://twitter.com/OptunaAutoML)
| [**LinkedIn**](https://www.linkedin.com/showcase/optuna/)
| [**Medium**](https://medium.com/optuna)


*Rustuna* is a faster Optuna implementation in Rust, featuring Python and JavaScript bindings.

## Why Rustuna?

The Optuna implementation in Rust is primarily motivated by two factors.

### Making Optuna Faster

![Rustuna vs Optuna: Speed Comparison](./docs/docs/assets/images/why-rustuna-1.jpg)

Optuna primarily targets hyperparameter optimization in machine learning. In such scenarios, model training and evaluation are typically time-consuming processes, so Optuna’s execution time seldom becomes the bottleneck.
However, black-box optimization has potential applications far beyond machine learning hyperparameter tuning. Rustuna is designed with such broader use cases in mind, including large-scale optimization workloads that may involve tens of thousands of trials or more.

### Broadening Language Support

The Rust-based implementation can be used not only from Rust and Python, but also through JavaScript bindings, which are used in the development of [Optuna Dashboard](https://github.com/optuna/optuna-dashboard).
Looking ahead, we are also interested in exploring support for additional languages.

## Design Philosophy

### Balancing Compatibility with Optuna and Performance

Rustuna provides essentially the same API as Optuna.
Users do not need to learn a new API, and they can transition straightforwardly from existing projects while continuing to benefit from the ecosystem that Optuna has built.
For example, the following code works as is by simply changing the import statement.

```python
import rustuna as optuna

def objective(trial: optuna.Trial) -> float:
    x = trial.suggest_float("x", -10, 10)
    y = trial.suggest_float("y", -10, 10)
    value = (x - 2) ** 2 + (y + 5) ** 2
    return value

study = optuna.create_study()
study.optimize(objective, n_trials=1000)
print(study.best_trial)
```

That said, Rustuna does not maintain this level of compatibility in every case.
Prioritizing compatibility too heavily would limit the performance improvements Rustuna can achieve.
For example, some features, such as `study.enqueue_trial()`, are implemented differently from Optuna’s and therefore cannot always be used in exactly the same way.
This is why we say that Rustuna provides “essentially” the same API as Optuna.

### Not Aiming to Completely Replace Optuna

Rustuna is not intended to serve as a complete replacement for Optuna.
Instead of re-engineering all of Optuna’s features in Rust, we aim to build a library that works alongside Optuna so that the strengths of each can complement one another.

For example, the [optuna.visualization module](https://optuna.readthedocs.io/en/stable/reference/visualization/index.html) provides rich functionality for visualizing and analyzing Optuna study results.
We do not plan to reimplement these features in Rust using libraries such as Plotly or Matplotlib. Instead, we aim to provide ways to use Rustuna’s results with the `optuna.visualization` module or with [Optuna Dashboard](https://github.com/optuna/optuna-dashboard).

## Installation 

### Python

> [!NOTE]
> Rustuna is currently experimental. Compared with Optuna, it still lacks some features and APIs, and it has not yet been optimized enough to deliver better performance for every use case. Since the project has not had the same level of maturity as Optuna, bugs and rough edges likely remain. We appreciate your understanding when using it.
> 
> If you are excited to try new software, we would love for you to give Rustuna a try and share your feedback with us.

You can install Rustuna via pip. Unlike Optuna, Rustuna doesn't have runtime dependencies, not even on NumPy. This not only eliminates concerns of version conflicts for users but also significantly speeds up imports.

```
$ pip install rustuna
```

Ready to try Rustuna in practice? The best place to start is the [Getting Started](https://rustuna.readthedocs.io/en/latest/tutorial/getting-started/) guide, which walks through the basic functionality and core APIs.

## Citation

If you use Rustuna for your research, please cite it using the following BibTeX entry:

```
@misc{rustuna,
  author = {Optuna Developers},
  title = {Rustuna: A faster Optuna implementation in Rust},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/optuna/rustuna}}
}
```

## License

MIT License
