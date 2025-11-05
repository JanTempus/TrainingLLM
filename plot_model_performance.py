import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import os

def plot_tokenizer_performance(vocab_sizes_1, performance_1,
                               vocab_sizes_2, performance_2,
                               label_1="Tokenizer A", label_2="Tokenizer B",
                               metric_name="Bits per Byte",
                               save_path="tokenizer_comparison.png",
                               show_plot=True):
    """
    Plot and save performance of two tokenizers across vocabulary sizes.

    Parameters
    ----------
    vocab_sizes_1, vocab_sizes_2 : list[int]
        Vocabulary sizes for tokenizer 1 and tokenizer 2.
    performance_1, performance_2 : list[float]
        Performance metric values for tokenizer 1 and tokenizer 2.
    label_1, label_2 : str, optional
        Labels for the two tokenizers.
    metric_name : str, optional
        Name of the metric (e.g., "Bits per Byte", "Loss", "Perplexity").
    save_path : str, optional
        File path where the PNG will be saved.
    show_plot : bool, optional
        Whether to also display the plot interactively.
    """

    plt.figure(figsize=(8, 5))
    plt.plot(vocab_sizes_1, performance_1, "o-", label=label_1, linewidth=2, markersize=6)
    plt.plot(vocab_sizes_2, performance_2, "s--", label=label_2, linewidth=2, markersize=6)

    plt.title(f"{metric_name} vs Vocabulary Size", fontsize=14, fontweight="bold")
    plt.xlabel("Vocabulary Size", fontsize=12)
    plt.ylabel(metric_name, fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()

    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    plt.savefig(save_path, dpi=300)
    print(f"✅ Plot saved as: {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()



if __name__ == "__main__":

    vocab_sizes    =[8196,32768,131072]
    lp_performance =[2.7507,3.1569,3.3036]
    bpe_performance=[2.6759,3.1207,3.3004]
    plot_tokenizer_performance(vocab_sizes, lp_performance, vocab_sizes, bpe_performance,
                           label_1="LP Tokenizer",
                           label_2="Baseline BPE",
                           metric_name="Perplexity")
