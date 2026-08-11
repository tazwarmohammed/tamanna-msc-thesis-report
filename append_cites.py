citations = """
@article{chen2022wavlm,
  title={WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing},
  author={Chen, Sanyuan and Wang, Chengyi and Chen, Zhengyang and Wu, Yu and Liu, Shujie and Chen, Zhuo and Li, Jinyu and Kanda, Naoyuki and Yoshioka, Takuya and Xiao, Xiong and others},
  journal={IEEE Journal of Selected Topics in Signal Processing},
  volume={16},
  number={6},
  pages={1505--1518},
  year={2022},
  publisher={IEEE}
}

@article{radford2023robust,
  title={Robust speech recognition via large-scale weak supervision},
  author={Radford, Alec and Kim, Jong Wook and Xu, Tao and Brockman, Greg and McLeavey, Christine and Sutskever, Ilya},
  journal={In International Conference on Machine Learning (ICML)},
  year={2023}
}

@inproceedings{desplanques2020ecapa,
  title={ECAPA-TDNN: Emphasized Channel Attention, propagation and aggregation in TDNN based speaker verification},
  author={Desplanques, Brecht and Thienpondt, Jenthe and Demuynck, Kris},
  booktitle={Interspeech 2020},
  pages={3830--3834},
  year={2020}
}

@inproceedings{ma2023emotion2vec,
  title={emotion2vec: Self-Supervised Pre-Training for Speech Emotion Representation},
  author={Ma, Ziyang and Zheng, Zhisheng and Ye, Jiaxin and Li, Jinchao and Gao, Zhikang and Zhang, Shiliang and Chen, Xie},
  booktitle={Interspeech 2024},
  year={2024}
}

@article{defossez2024moshi,
  title={Moshi: a speech-text foundation model for real-time dialogue},
  author={Defossez, Alexandre and Copet, Jade and Synnaeve, Gabriel and Roziere, Baptiste},
  journal={arXiv preprint arXiv:2410.00037},
  year={2024}
}
"""

with open('bibliography/references.bib', 'a') as f:
    f.write(citations)
