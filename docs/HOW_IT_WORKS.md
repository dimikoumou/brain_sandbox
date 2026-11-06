# How it works, in plain language

## The idea in one paragraph

Brain scientists learn a lot from people who have damage to one part of the
brain: if someone loses one ability but keeps another, those two abilities
probably rely on different machinery. We can't damage real brains on purpose,
but we *can* damage a computer model as often as we like. So we built a small
artificial "brain", taught it ten simple games, and then broke it in many
different ways to see which games stop working.

## What the model is

* It is made of **256 simple units**. Each unit is a number (its "activity")
  that goes up when other units excite it and down when they inhibit it.
  Think of each unit as standing for a small group of brain cells.
* The units are split into **four regions**, wired together like this:

  ```
     sights & sounds ──► SENSORY ◄──► MEMORY
                           ▲  │          ▲
           feedback ──────►│  ▼          │
                         CONCEPTUAL ◄──► EXECUTIVE ◄── which game
                                          │
                                          ▼
                                        answer
  ```

  * **sensory** receives everything the model sees and hears;
  * **memory** sits between sensory and executive (we *hoped* it would
    hold things in mind);
  * **conceptual** is a "big picture" region that also sends feedback down to
    sensory;
  * **executive** is told which game is being played and gives the answer.

  These names describe *where each region sits in the wiring*. Whether a
  region really does what its name says is something we measure, not assume.
* Like real cells, every unit is either **excitatory** (it pushes others up)
  or **inhibitory** (it calms others down), never both. 80% are excitatory.
* The model was **trained**: we showed it thousands of example trials and
  nudged its connections a little after each mistake, until it played every
  game well.

## The games

The model looks at a "screen" with lights in different directions, hears
tones, and is told which game to play.
It answers by "looking" in a direction, or by pressing one of two buttons.
In every game it must wait (keep looking at the centre) until a go signal.

| Game | What the model has to do |
|---|---|
| Look at the light | look where a light appears |
| Look away from the light | look in the opposite direction |
| Remember where the light was | a light flashes, disappears; after a wait, look where it was |
| Pick the brighter light (channel 1 / 2) | two lights; look at the brighter one |
| Pick the brighter light, ignoring channel 2 / 1 | lights on two channels; only one channel counts |
| Same place twice? Then look | two lights, one after another: look only if they were in the same place |
| Different place? Then look | the opposite rule |
| Did the tune go up or down? | two notes; press button 1 if the second is higher, button 2 if lower |

The last game is about **relative pitch**: the first note can be any note, so
knowing *which* notes were played (absolute pitch) doesn't help; only the
direction of the step matters.

## What we can do to it (the knobs)

| Knob | What it means |
|---|---|
| Switch off units | silence some or all units of a region (like a stroke) |
| Cut a connection | remove the wiring from one region to another |
| Noise | make a region's units jittery |
| Excitability (gain) | make units more or less responsive (like some drugs) |
| Inhibition | weaken or strengthen all the "calming" connections |
| Speed (time constant) | make a region react more slowly or quickly |
| Delete weak connections | remove the weakest X% of all connections |
| Rehabilitation | after damage, let the model practise again and see what comes back |

None of these knobs retrains the model (except rehabilitation): we take the
trained model, change it, and test it again on exactly the same trials.

## Hidden absolute pitch

We play single notes and then ask: **after the note has stopped**, does the
activity in each region still say *which* note it was? We test this with a very
simple decoder (a straight-line rule, logistic regression) that tries to name
the note (C, C#, D, ... B) from the activity. If it can only guess, it is right
1 time in 12 (8%).

This matters for an idea from brain science (Allan Snyder's "release"
hypothesis): some people with savant skills, such as perfect pitch, may have
access to raw detail that the rest of us also process but normally discard,
because "big picture" brain areas take over. If that were true in our model,
weakening the conceptual region should make *more* note information
readable in sensory.

## How we keep ourselves honest

* **Many copies.** Every result is repeated on several copies of the model
  (5 for the main model), each trained from a different random start ("seeds").
  A result that appears in one copy only is not a result.
* **Error bars.** Numbers come with a 95% confidence interval: the range
  where the average over many copies would probably fall.
* **Correction for many tests.** When we look at hundreds of combinations, some
  look interesting by pure chance. We use a standard correction (Holm) so
  that luck is not reported as a discovery, and we ignore effects smaller than
  10 percentage points.
* **Predictions written first.** Each experiment file contains a prediction
  written *before* running it; the report shows prediction vs. outcome,
  including when the prediction was wrong.
* **Controls.** For hidden pitch we also measure (1) a version of the
  model before any training, (2) the same activity with the note labels
  shuffled (what pure guessing scores), and (3) the raw input.

## What this model can't tell us

* It is tiny (256 units versus billions of neurons) and learns in a way real
  brains don't (a mathematical shortcut called backpropagation).
* It has no body, no development from childhood, no sleep, no chemistry
  beyond one "excitability" knob, no glia (the brain's support cells).
* Region names are labels for wiring positions, nothing more.

So results here are **ideas to test**, never facts about people.
