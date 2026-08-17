# 1. Domain and dataset — explained simply

## 1.1 What kind of business problem is this?

Think of a factory that makes parts. Its production machine has a rotating spindle and a cutting tool. While it works, its operating condition can be described by temperature, rotational speed, torque and tool wear.

If the operating point becomes unhealthy, the machine may stop or produce poor parts. Unexpected stoppage costs money and interrupts production. A maintenance engineer therefore wants early, explainable evidence—not just a red light.

The AI4I dataset is a **synthetic milling-machine-style predictive-maintenance benchmark**. It represents a production-machine situation, but it is not data from an identified real factory or a live physical machine.

## 1.2 What is one row of data?

One row is one synthetic observation of a machine operating condition. It contains its operating readings plus labels telling us whether the benchmark marked a failure.

There are **10,000 rows** and **14 source columns**.

| Data check | Result |
|---|---:|
| Rows | 10,000 |
| Source columns | 14 |
| Missing values | 0 |
| Exact duplicate rows | 0 |
| Machine-failure rows | 339 |
| Overall failure rate | 3.39% |

### Did we remove noise, records, or columns?

No raw observation was removed, smoothed, or imputed during loading. The loader performs an exact schema check and then reads the original CSV. `UDI` is only renamed to `UID` **inside the application** to provide a conventional unique-identifier name. The source CSV itself is not changed.

This is important: there was no arbitrary “noise removal.” The project adds calculated features in a copied working table, while preserving the raw data.

## 1.3 Every source column

| Column | What it means | Why it matters |
|---|---|---|
| `UDI` / `UID` | Unique observation number. `UDI` is the source name; the app calls it `UID`. | Identifies a row; not used to predict risk. |
| `Product ID` | Identifier for the product/work item. | Helpful reference information; not used as a risk feature. |
| `Type` | Product quality/variant: `L`, `M`, or `H`. | Changes the overstrain threshold, so it is useful to the model. |
| `Air temperature [K]` | Temperature of surrounding air in Kelvin. | Used with process temperature to calculate temperature difference. |
| `Process temperature [K]` | Temperature near/in the production process in Kelvin. | Used with air temperature to check thermal conditions. |
| `Rotational speed [rpm]` | How fast the spindle rotates: revolutions per minute. | Affects power and the HDF rule. |
| `Torque [Nm]` | Turning force applied by the rotating system, in Newton-metres. | Affects power and overstrain load. |
| `Tool wear [min]` | Accumulated cutting-tool use/wear, measured in minutes in this benchmark. | More wear can increase overstrain load. |
| `Machine failure` | Main yes/no outcome label: `1` means benchmark failure, `0` means no benchmark failure. | The target the ML model learns to estimate. |
| `TWF` | Tool Wear Failure flag. | Failure-mode label for analysis; excluded from model input to prevent cheating. |
| `HDF` | Heat Dissipation Failure flag. | Failure-mode label for analysis; excluded from model input. |
| `PWF` | Power Failure flag. | Failure-mode label for analysis; excluded from model input. |
| `OSF` | OverStrain Failure flag. | Failure-mode label for analysis; excluded from model input. |
| `RNF` | Random Failure flag. | Failure-mode label for analysis; excluded from model input. |

### Units without jargon

- **K (Kelvin):** temperature scale. A temperature difference of 1 K equals a difference of 1°C.
- **rpm:** number of full turns per minute.
- **Nm:** rotational force. More torque means more twisting force.
- **W (Watt):** mechanical power: how quickly work is being done.
- **min Nm:** the project’s overstrain-load unit, obtained by multiplying tool wear minutes by torque.

## 1.4 What values occur in this dataset?

| Measurement | Minimum | Average | Maximum |
|---|---:|---:|---:|
| Air temperature | 295.3 K | 300.0 K | 304.5 K |
| Process temperature | 305.7 K | 310.0 K | 313.8 K |
| Rotational speed | 1,168 rpm | 1,538.8 rpm | 2,886 rpm |
| Torque | 3.8 Nm | 40.0 Nm | 76.6 Nm |
| Tool wear | 0 min | 108.0 min | 253 min |

## 1.5 Product types and failures

| Type | Records | Machine failures | Failure rate |
|---|---:|---:|---:|
| L | 6,000 | 235 | 3.92% |
| M | 2,997 | 83 | 2.77% |
| H | 1,003 | 21 | 2.09% |

`L`, `M`, and `H` are benchmark product categories, not a human grade of “bad machine.” In the documented overstrain rule, they have different permitted thresholds.

## 1.6 Failure modes: an easy caution

The dataset contains these counts: TWF 46, HDF 115, PWF 95, OSF 98, RNF 19.

Do **not** add them and expect exactly 339. One observation can carry more than one failure-mode flag, so the labels can overlap. The project never uses these failure flags as model input because that would leak the answer the model is supposed to predict.

## 1.7 Why 3.39% is important

Failures are rare: 339 out of 10,000 records. A silly model that always says “no failure” would appear about 96.61% accurate, while being useless at finding failures. That is why this project focuses on PR-AUC, ROC-AUC, Brier score, recall and precision—not plain accuracy.
