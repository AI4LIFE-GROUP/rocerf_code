# Towards Bridging the Gaps between the Right to Explanation and the Right to be Forgotten

Abstract : The Right to Explanation and the Right to be Forgotten are two important principles outlined to regulate algorithmic decision making and data usage in real-world applications. While the right to explanation allows individuals to request an actionable explanation for an algorithmic decision, the right to be forgotten grants them the right to ask for their data to be deleted from all the databases and models of an organization. Intuitively, enforcing the right to be forgotten may trigger model updates which in turn invalidate previously provided explanations, thus violating the right to explanation. In this work, we investigate the technical implications arising due to the interference between the two aforementioned regulatory principles, and propose the first algorithmic framework to resolve the tension between them. To this end, we formulate a novel optimization problem to generate explanations that are robust to model updates due to the removal of training data instances by data deletion requests. We then derive an efficient approximation algorithm to handle the combinatorial complexity of this optimization problem. We theoretically demonstrate that our method generates explanations that are provably robust to worst-case data deletion requests with bounded costs in case of linear models and certain classes of non-linear models. Extensive experimentation with real-world datasets demonstrates the efficacy of the proposed framework.

## Cite Us

If you find our work useful, cite our paper:

```
@inproceedings{Krishna2023TowardsBT,
  title={Towards Bridging the Gaps between the Right to Explanation and the Right to be Forgotten},
  author={Satyapriya Krishna and Jiaqi Ma and Himabindu Lakkaraju},
  booktitle={International Conference on Machine Learning},
  year={2023},
  url={https://api.semanticscholar.org/CorpusID:256697597}
}
```

