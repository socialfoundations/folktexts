.. folktexts documentation master file, created by
   sphinx-quickstart on Wed Dec 13 15:35:43 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. error-parity documentation master file, created by
   sphinx-quickstart on Thu Nov 23 16:53:42 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


Welcome to :code:`folktexts`' documentation!
============================================

The :code:`folktexts` package enables you to benchmark and evaluate 
LLM-generated risk scores.

We encode unrealizable tabular prediction tasks as natural language text tasks,
and prompt LLMs for the probability of a target variable being true.
The correct solutions for each task often require expressing uncertainty, as the
target variable is not uniquely determined by the input features.

Folktexts is compatible with any huggingface transformer model and models 
available through web APIs (e.g., OpenAI API).

Five tabular data tasks are provided out-of-the-box, using the American
Community Survey as a data source: `ACSIncome`, `ACSMobility`, `ACSTravelTime`, 
`ACSEmployment`, and `ACSPublicCoverage`. These tasks follow the same name, 
feature columns, and target columns as those put forth by `Ding et al. (2021)`_ 
in the `folktables`_ python package.


Full code available on the `GitHub repository`_, 
including various `jupyter notebook examples`_ .

Check out the following sub-pages:

.. toctree::
   :maxdepth: 1

   Readme file <readme>
   API reference <source/modules>
   Example notebooks <notebooks>


Indices
=======

* :ref:`genindex`
* :ref:`modindex`


.. _folktables: https://github.com/socialfoundations/folktables
.. _Ding et al. (2021): https://arxiv.org/abs/2108.04884
.. _GitHub repository: https://github.com/socialfoundations/folktexts
.. _jupyter notebook examples: https://github.com/socialfoundations/folktexts/tree/main/notebooks
