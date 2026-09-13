"""The Video-HopChain dataset pipeline, one module per stage.

    proxy       stage 1   admit a video of at least three minutes and write its frame proxy
    segment     stage 2   split the video into shots
    caption     stage 3   caption every shot
    spec        stage 4   draw the specification of every question
    generate    stage 5   fill the specification from the captions
    checks      stage 6   recompute every answer in code and drop a candidate whose arithmetic fails
    judge       stage 7   re-examine every hop against the captions
    generate    stage 8   write a faulted question again, until the judge passes it
    difficulty  stage 9   drop the questions the base model already solves
    assemble    stage 10  render the rows and split by video

`prompts` holds the four prompts, `model` the client every model stage calls through, and
`schema` the hop record and the released row format.
"""
