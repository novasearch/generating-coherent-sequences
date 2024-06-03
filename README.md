# Generating Coherent Sequences of Visual Illustrations for Real-World Manual Tasks

This is the official repository for Generating Coherent Sequences of Visual Illustrations for Real-World Manual Tasks (ACL 2024).

## Code Structure

`coherence-annotations/` - code for AMT and analysis

`data/` - dataset used for training and evaluating

`gen-images/` - code for generating images given prompt files

`generation/` - main code for generation with the latents

`LAVIS/` - code for captioning, requires cloning InstructBLIP, see `LAVIS/README.md`

`llm-caption-finetuning/` - files and scripts for creating the training data and code for automatic metric

`models/` - model LoRA checkpoint for Sequence Context Decoder

`notebooks/` - useful notebooks

`PlanGPT/` - code for training

`recipe-filtering-analysis/` - code for filtering and analysing recipes

## Citation
If you find this work useful, please cite using the following BibTeX:
```
@misc{bordalo2024generating,
      title={Generating Coherent Sequences of Visual Illustrations Real-World Manual Tasks}, 
      author={João Bordalo and Vasco Ramos and Rodrigo Valério and Diogo Glória-Silva and Yonatan Bitton and Michal Yarom and Idan Szpektor and Joao Magalhaes},
      year={2024},
      eprint={2405.10122},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```