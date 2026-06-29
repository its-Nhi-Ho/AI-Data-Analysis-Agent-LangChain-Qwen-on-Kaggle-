# 🤖 AI Data Analysis Agent (LangChain + Qwen on Kaggle)

> A multi-agent data analysis system built with LangChain and Qwen2.5, adapted from an original implementation using GPT-4o-mini and Google ADK — rewritten to run fully free on Kaggle GPUs.

---

## 📌 Background

This project is a fork/adaptation of an existing multi-agent data analysis repo that used **GPT-4o-mini** as the LLM backbone and **Google ADK** as the agent orchestration framework.

To eliminate API costs and make the system fully self-contained, this version replaces those components with:

| Original | This Repo |
|---|---|
| GPT-4o-mini (paid API) | Qwen2.5-3B / 7B-Instruct (free, local) |
| Google ADK | LangChain ReAct Agent |
| Cloud execution | Kaggle Notebook (free T4 GPU) |

---

## 🏗️ Architecture

The system uses a **hierarchical multi-agent** design with a coordinator agent delegating to specialized sub-agents:

```
analyze(prompt)
    └── Orchestrator Agent (Qwen2.5)
            ├── data_loader_agent     → load_csv, create_sample_dataset
            ├── statistician_agent    → describe_dataset, correlation_analysis,
            │                           hypothesis_test, outlier_detection
            ├── visualizer_agent      → create_visualization, create_distribution_report
            ├── transformer_agent     → filter_data, aggregate_data
            └── reporter_agent        → generate_summary_report
```

All datasets are shared via a singleton `DataStore` across agents in the same session.

---

## 🛠️ Tools Available

| Tool | Description |
|---|---|
| `create_sample_dataset` | Generate synthetic sales/customer/timeseries/survey data |
| `load_csv` | Load a CSV file from disk into the DataStore |
| `list_available_datasets` | List all active datasets |
| `describe_dataset` | Full descriptive statistics (df.describe) |
| `correlation_analysis` | Pearson/Spearman correlation matrix |
| `hypothesis_test` | Shapiro-Wilk, t-test, ANOVA, Chi-square |
| `outlier_detection` | IQR or Z-score outlier detection |
| `create_visualization` | histogram, scatter, bar, line, box, heatmap, pie |
| `create_distribution_report` | 4-panel distribution report (hist + box + QQ + violin) |
| `filter_data` | Filter rows via pandas query syntax |
| `aggregate_data` | Group-by aggregation |

---

## ⚡ Quickstart (Kaggle)

1. Add your dataset to the Kaggle notebook via **Add Data**
2. Run all setup cells (install deps, load model)
3. Load your data manually (recommended — see Known Issues):

```python
import pandas as pd
df = pd.read_csv("/kaggle/input/your-dataset/file.csv")
store.add("my_data", df)
```

4. Start analyzing:

```python
print(analyze("Describe 'my_data' và vẽ histogram của revenue"))
```

---

## 🔬 Model Comparison: Qwen2.5-3B vs 7B

Both models were tested on the same multi-agent task set. Findings from actual run logs:

| Behavior | Qwen2.5-3B | Qwen2.5-7B |
|---|---|---|
| Basic tool call (`describe`, `correlation`) | ✅ Works | ✅ Works |
| Visualization (`histogram`) | ⚠️ Works after prompt fix | ✅ Works |
| Dataset name hallucination (`df_fifa`) | ✅ Consistent pattern | ✅ Same pattern |
| CSV path in Action Input | ❌ Always drops/hallucinates path | ❌ Same failure |
| Language discipline (Vietnamese prompt) | ✅ Stays in Vietnamese | ❌ **Switches to Chinese mid-run** |
| ReAct format compliance | ✅ Generally follows format | ❌ Frequently outputs Action + Final Answer simultaneously |
| Infinite retry loop | ⚠️ Occasionally repeats same failed action | ❌ More severe — loops 5-8x on same error |
| Hallucinated URLs in Final Answer | ❌ Rare | ❌ **Fabricates fake Kaggle/S3 image URLs** |
| `filter_data` with quoted strings | ❌ Fails (expr parse error) | Not tested |
| Overall tool routing accuracy | ~65% | ~60% (more format errors offset capability gain) |

**Verdict:** 7B is not strictly better for agentic tasks. Its larger context capacity causes it to generate more verbose and structurally invalid outputs, leading to more ReAct format violations. For this use case, **3B with tighter prompts is more predictable**.

---

## ✅ Strengths

**Cost-free execution**
Runs entirely on Kaggle's free T4 GPU — no OpenAI API key, no Google Cloud billing. Zero cost per run.

**Solid tool design pattern**
All multi-parameter tools use a unified `tool_input: str` + `_parse_kv_string()` pattern instead of StructuredTool, which is the correct approach for ReAct agents with small LLMs that generate flat `key=value` strings.

**Resilient key-value parser**
The custom `_parse_kv_string()` handles edge cases like values containing `=` signs (e.g. `condition=age >= 30`) — better than naive `split(",")`.

**Singleton DataStore**
Clean shared state across all sub-agents without passing dataframes through tool arguments.

**Graceful fallback on load failure**
When `load_csv` fails (file not found), the 3B model falls back to `create_sample_dataset` instead of crashing — unintended but practical behavior observed in logs.

---

## ⚠️ Known Issues (Observed from Logs)

### 🔴 Critical

**7B switches language to Chinese**
When the orchestrator delegates with a Chinese-language internal task string (e.g. `task=显示sales_data数据集的前5行`), sub-agents respond entirely in Chinese — ignoring the Vietnamese system prompt. Root cause: 7B model leaks its internal "thinking" language into output.

**7B fabricates image URLs in Final Answer**
After successfully creating a chart, the 7B model generates fake S3/Kaggle URLs pointing to non-existent files instead of using the actual saved path `/kaggle/working/last_chart.png`.

**ReAct format violation (7B)**
7B frequently outputs both `Action` and `Final Answer` in the same turn, breaking the parser. Observed 10+ times in a single session. The custom error message (`Lỗi format: bạn chỉ được viết MỘT Action...`) helps but does not fully resolve it.

### 🟡 Moderate

**`dataset_name=sales_data` passed as literal string**
Fixed in this repo — original tool used typed `dataset_name: str` param causing LangChain StructuredTool to pass `"dataset_name=sales_data"` verbatim. Now uses `tool_input: str` + `_parse_kv_string()`.

**`column` vs `x` key mismatch in histogram**
Model generates `column=revenue` but tool expected `x=revenue`. Fixed with `or args.get("column")` fallback.

**Dataset name prefix hallucination**
Both models add `df_` prefix to dataset names (e.g. `df_fifa` instead of `fifa`). Fixed by adding `name.removeprefix("df_")` fallback in `DataStore.get()`.

**Infinite retry without adaptation (3B)**
When `load_csv` fails, 3B retries the exact same action with the same bad path 3-4 times before giving up. No adaptive behavior.

**`filter_data` fails with quoted string values**
Pandas `.query()` rejects expressions like `condition=name == "John"` due to quote escaping issues. Observed multiple failed attempts in logs — no fix applied yet.

### 🟢 Minor

**`hypothesis_test` group mismatch not handled**
When user requests a t-test on a column with 3+ groups, tool correctly rejects it (`T-test cần đúng 2 nhóm`) but the agent doesn't suggest ANOVA as an alternative on its own.

**DataStore not persisted across Kaggle sessions**
All loaded datasets live in memory only. Kernel restart requires manual reload.

---

## 💡 Recommended Usage Pattern

```python
# ✅ DO: Load files manually before calling analyze()
df = pd.read_csv("/kaggle/input/.../file.csv")
store.add("fifa", df)  # use short names, no spaces

# ✅ DO: Give focused, single-task prompts
print(analyze("Vẽ histogram của overall_rating trong df 'fifa'"))

# ✅ DO: Prefer 3B over 7B for Vietnamese prompts
# 7B tends to switch to Chinese and generate format violations

# ❌ DON'T: Ask agent to load AND analyze in one prompt
print(analyze("Load file /kaggle/input/.../file.csv và phân tích nó"))

# ❌ DON'T: Use filter_data with string equality conditions
# condition=name == "East"  →  pandas query parse error
```

---

## 📦 Dependencies

```
transformers
langchain
langchain-community
langchain-huggingface
pandas
numpy
scipy
matplotlib
seaborn
```

---

## 🙏 Credits

Original architecture inspired by a multi-agent data analysis system built with **GPT-4o-mini** and **Google ADK**. This repo adapts that design to run on open-source models at zero cost.
