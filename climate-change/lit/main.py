import lit_nlp

datasets = {
    'foo_data': FooDataset('/path/to/foo.tsv'),
    'bar_data': BarDataset('/path/to/bar.tfrecord'),
}
models = {'my_model': MyModel('/path/to/model/files')}
lit_demo = lit_nlp.dev_server.Server(models, datasets, port=4321)
lit_demo.serve()