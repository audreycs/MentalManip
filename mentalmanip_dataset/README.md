:loudspeaker: **Announcement**: 

- [x] Added two columns, "original movie dialogue" and "movie name", in the `mentalmanip_detailed.csv` file.
- [x] Uploaded the datasets to the [Hugging Face Repo](https://huggingface.co/datasets/audreyeleven/MentalManip).

:heavy_exclamation_mark: **Note**:

When processing the data files, I suggest using `csv` instead of `pandas.read_csv` because `pandas` may not read the columns correctly. For example
```python
# read .csv files
with open("dialogue_detailed.csv", 'r', newline='', encoding='utf-8') as infile:
    content = csv.reader(infile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
    for idx, row in enumerate(content):
        ...

# write .csv files
with open("example.csv", 'w', newline='', encoding='utf-8') as outfile:
    writer = csv.writer(outfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
    for row in data:
        writer.writerow(row)
        ...
```

## Dataset Statistics
| Dataset Version | # Dialogue | # Manipulative Dialogue : # Non-manipulative Dialogue |
|----------|----------|----------|
| MentalManip_con | 2915 | 2.24:1 |
| MentalManip_maj | 4000 | 2.38:1 |

**Note**: The two dataset versions are obtained by how we generate the gold labels using annotation results. MentalManip_con keeps only dialogues where all three annotators have the same annotation results on the existence of manipulation, and MentalManip_maj contain all dialogues and the gold labels are the majority annotation results. Fpr more details about final label generation, please see the paper.

## File Description

### 1) mentalManip_detailed.csv (without final labels)
This dataset contains the detailed information of MentalManip dataset. 
Each row contains one dialogue and its three annotors' annotations.
The columns are:
- `inner_id`: inner id of the dialogue, from 0 to 3999.
- `id`: unique id string of the dialogue for identification.
- `dialogue`: the dialogue text.
- `original movie dialogue`: the orignal movie dialogue in Cornell Movie-Dialogs Corpus that this dialogue is based on.
- `movie name`: the name of the movie from which the orignal movie dialogue is extracted.
- `agreement`: the agreement of the three annotors.
- `annotator_1`: the id of annotator 1 (e.g. AN12).
- `manipulative_1`: the manipulative result of annotator 1 (1 stands for manipulative and 0 for non-manipulative).
- `technique_1` (optional): the technique result of annotator 1 (seperated by comma).
- `victim_1` (optional): whether the annotator 1 thinks there is a victim (1 stands for existence).
- `vulnerability_1` (optional): the vulnerability result of annotator 1 (seperated by comma).
- `marks_1` (optional): the manipulative parts marked by annotator 1.
- `confidence_1`: the confidence score of annotator 1 (1 to 5).
- (following columns are similar for annotator 2 and 3)

### 2) mentalmanip_con.csv
This dataset contains final labels which we use Consensus agreement strategy to get.

> **Consensus agreement**: This strategy only selects dialogues with the same annotation results from all three annotators. The accordant result becomes the final label.

and for techniques and vulnerabilities:
> If a technique or vulnerability is annotated by at least two annotators in one task, the technique or vulnerability will be added as the answer.

The columns in `mentalmanip_con.csv` are:
- `ID`: unique id string of the dialogue for identification.
- `Dialogue`: the dialogue text.
- `Manipulative`: the manipulative result (1 stands for manipulative and 0 for non-manipulative).
- `Technique`: the technique result (seperated by comma).
- `Vulnerability`: the vulnerability result (seperated by comma).

### 3) mentalmanip_maj.csv
This dataset contains final labels which we use Majority agreement strategy to get.

> **Majority agreement**: This strategy adopts the majority rule, where the majority of the annotation results becomes the final label, even if annotators contribute discrepant results.

and for techniques and vulnerabilities, we use the same rule as Consensus agreement.

The columns in `mentalmanip_maj.csv` are also the same as `mentalmanip_con.csv`.
