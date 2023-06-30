<div align="center">

# Omneer SDK

Omneer SDK is an innovative toolkit designed for building AI and machine learning driven personalized medicine applications. It allows developers to effortlessly construct, deploy, and interface with such applications using Python.

Omneer SDK leverages Argo, a powerful, Kubernetes-native workflow engine to deliver robust scalability and resilience. It guarantees task-level atomicity and offers containerization, independent task scheduling, and versatile, scalable computing resources.

[Discord Community Community]() • [Docs](https://docs.omneer.xyz) • [Installation](#installation) •
[Quickstart](#configuration) • [Omneer](https://omneer.xyz)

</div>

Workflows developed with the SDK feature:

  * Instant no-code interfaces for accessibility and publication
  * First class static typing
  * Containerization + versioning of every registered change
  * Reliable + scalable managed cloud infrastructure
  * Single line definition of arbitrary resource requirements (eg. CPU, GPU) for serverless execution

Utilizing the Omneer SDK, developers can deliver:

- Personalized AI and machine learning tools
- Advanced disease progression diagnosis and tracking
- Creation of digital twins for individualized healthcare
- Instant no-code interfaces for rapid deployment
- High-performing, reliable cloud infrastructure
- Flexibility to define resources (CPU, GPU, etc.) for serverless execution

- Omneer SDK continues to be a ground-breaking platform for personalized healthcare solutions. Browse our collection of existing and actively maintained solutions at [Omneer Community]().

### Getting Started

See the SDK in action by following the steps below to register your first workflow with Omneer.

First, install latch through `pip`.

```
$ python3 pip install omneer
```

Then, create some boilerplate code for your new workflow.

```
$ omneer init diagnosis
```

The registration process, which could take a few minutes depending on your network connection, involves building a Docker image with your workflow code, serializing the code, registering it with your Omneer account, and pushing the Docker image to a managed container registry.

Upon successful registration, your new workflow should be visible in your Omneer Console.

For issues with registration or other queries, please raise an issue on GitHub.

---

### Installation

Omneer SDK is distributed via pip. We recommend installing it in a clean virtual environment for the best user experience.

Virtualenv is our recommended tool for creating isolated Python environments.

[Virtualenvwrapper](https://virtualenvwrapper.readthedocs.io/en/latest/) is recommended.

```
pip install omneer
```

### Examples

[Omneer Examples]() features list of well-curated workflows developed by the Omneer team. 
* [Parkinson's Diagnosis]()
* [Parkinson's Progression]()
* [Parkinson's Personalized]()

We'll maintain a growing list of well documented examples developed by our community members here. Please open a pull request to feature your own:

**Parkinson's Diagnosis**
  * [Early Detection]()

 
  
