from pymlpipe import pipeline

ppl=pipeline.PipeLine("IrisData")
ppl.load_pipeline()
ppl.run()
