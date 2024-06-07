## Datasets Description

### 1) mentalManip_detailed.csv
This dataset contains the detailed information of MentalManip dataset. 
Each row contains one dialogue and its three annotors' annotations.
The columns are:
- `inner_id`: inner id of the dialogue, from 0 to 3999.
- `id`: unique id string of the dialogue for identification.
- `dialogue`: the dialogue text.
- `agreement`: the agreement of the three annotors.
- `annotator_1`: the id of annotator 1 (e.g. AN12).
- `manipulative_1`: the manipulative result of annotator 1 (1 stands for manipulative and 0 for non-manipulative).
- `technique_1`: the technique result of annotator 1 (seperated by comma).
- `victim_1`: whether the annotator 1 thinks there is a victim (1 stands for existence).
- `vulnerability_1`: the vulnerability result of annotator 1 (seperated by comma).
- `marks_1`: the manipulative parts marked by annotator 1.
- `confidence_1`: the confidence score of annotator 1 (1 to 5).
- (following columns are similar to annotator 1)

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
