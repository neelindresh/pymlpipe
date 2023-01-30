from pymlpipe import pipeline

ppl=pipeline.Pipeline("IrisData")
ppl.load_pipeline()
ppl.run()
