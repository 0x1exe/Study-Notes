# System Design

# ML System Design

MACHINE LEARNING SYSTEM DESIGN is a complex, multistep process of designing, implementing,
and maintaining machine learning-based systems that involves a combination of techniques and
skills from various fields and roles, including machine learning, deep learning, software
engineering, project management, product management, and leadership.

Each subchapter is a high-level checklist item mandatory for every ML system. Note:
while not all the items must be fulfilled, each of them must be remembered and
considered.

On top of that, each subchapter should answer the question why and when the given
item is important. It also should include a description of the landscape (what techniques
and tools are suitable for the item). The description must be systematized (not just a list
of a hundred buzzwords), although not necessarily exhaustive, as we believe that an
experienced reader will be able to compare the example case with an issue from their
background and draw their own conclusions.

## 1.3 When principles of ML system design can be helpful

As we said earlier, applying these principles is critical to build a system complex enough
to have multiple failure modes. Ignoring them leads to high chances of delivering
something with feet of clay—a system that may work right now but is not sustainable to
survive a challenge from the dynamic environment of reality. 
**The challenge can be purely technical (what if we face ten times more data?), product-related (how do we
adapt for changed user scenarios?), business-driven (what if the system is to be included
into third-party software stack after an acquisition?), legal (what if the government
issues a new regulation on personal data management?) or anything else. Recent years
have only proved we can’t foresee every possible risk.**

Improving the system is even more important. As we’ll describe with more details in
the upcoming chapters, building systems from scratch is a relatively rare event. People
out of the industry may think software engineers spend most of their time writing code,
while in reality, as we all know, way more time is dedicated to reading code. Same goes
with systems: much more efforts are usually spent improving and maintaining existing
systems, not building them for scratch.

The sad truth is that very often systems are maintained by teams who didn’t participate in
building them. So, it’s a two-side blade: the building team should keep some principles in
mind to simplify lives of their followers, and the maintaining team should understand the
principles to be able to understand the whole system logic fast enough and find proper
workarounds to keep the system alive over a long period of time.

**It is safe to say that close to 100% of ML projects which hadn’t had a well-written
design document failed, whereas a sweeping majority of those systems that had been
thoroughly planned found success. The design document in this case plays two major
roles. Not only does it set proper priorities within a project, it also helps understand
whether you actually need this project in the first place and drags your gaze away from
the core idea (you might be too focused on the project itself) to seeing the whole
picture.**

## 2. Is there a problem? 

What are the business goals? How big is the budget? How flexible are the deadlines?
Will the potential output cover and exceed overall costs? These are among the crucial
questions that you need to ask yourself before scoping your ML project.

**But before you start addressing these questions, there is a paramount action that will
lay the foundation for successful ML system design, and it’s finding and articulating the
problem your solution will solve (or help solve). A seemingly trivial point, especially for
experienced engineers, but based on our own practice in the area, skipping this step in
your preliminary work is deceptively dangerous, and this is what we will try to cover in
this chapter.**

### 2.1 Problem space vs. solution space

- **While thinking and asking questions, he was focused on the solution space, not the problem space**

What are the problem space and solution space? These are two exploration paradigms
that cover different perspectives of a problem. While both are crucial, the former should
always precede the latter.

The problem space is often defined with What and Why questions, often even with
chains of such questions. There is even a popular technique named “Five Whys” that
recommends stacking your Why questions on top of each other to dig to the very origin
of the problem you analyze. Typical questions often look like this:

	1.Why do we need to build the solution?
	2.What problem does it solve?
	3.Why does the problem occur?
	4.What are the alternatives we know?
	5.Why do we want to make it work with given limitations (metrics, latency, number of training samples)?

![[Pasted image 20240529155932.png]]
- **An experienced engineer always handles the problem space first with specifying questions**

The What part, in its turn, is about understanding the customer and functional
attributes. E.g., “A tool that annotates customer leads with a score showing how likely the
deal will happen; it should assign the scores before sales managers plan their work on a
Monday weekly meeting."
![[Pasted image 20240529160043.png]]
The solution space is somewhat opposite. It’s less about the problem and customer needs, and more about the implementation. Here, we talk about frameworks and interfaces, discuss how things work under the hood, and consider technical risks. However, it should never be done before we reach a consistent understanding of a problem.

```
Reaching a solid understanding before thinking on a technical implementation allows
you to consider various workarounds, some of which may significantly reduce the project
scope. Maybe there is a third-party plugin for the CRM that is designed to solve exactly
this problem? Maybe the cost of error for the ML part of such a problem is not really that
important despite the first Jack’s answer (stakeholders start with the statement they
need accuracy close to 100% so often!)? Maybe the data shows that 95% of empty leads
can be filtered out with simple rule-based heuristics? We don’t know. What we do know
is, a vanishingly rare successful ML industry project starts with drafting APIs and reading
state-of-the-art papers at the earliest stage.
```

### 2.2 Finding the problem

``` MelvinE.Conway
“Organizations which design systems (in the broad sense used here) are
constrained to produce designs which are copies of the communication structures of
these organizations.”
```

We encourage you to write down a problem statement using a reverted pyramid
scheme with a high-level understanding in its basement and nuances at the top. 
![[Pasted image 20240529160635.png]]
**On the very top level**, you can formulate the helicopter-view understanding of the
problem. That's the level understandable to any C-level office of the organization where
people don't care too much about ML algorithms or software architecture. E.g.:

	1. There are fraudsters in our mobile app who try to attack our legit users
	2. Our pricing model demonstrated extremely low margin profits for some product while being absolutely uncompetitive in other categories
	3. Customers complain that our software requires a lot of manial tuning before bringing value

Having such a statement at the start gives many opportunities for the next exploration steps. Just try to question every word in a given sentence to make sure you can explain it to a ten-year-old child. Who are fraudsters? How do they attack? What report gave the initial insight about excessive prices? What bothers our customers the most? Where is the
most time wasted? How do we measure user engagement? How are recommendations related to this metric? Ask yourself or your colleagues questions until you’re ready to build the next, broader block of the pyramid that expands the initial one

**This next pyramid block requires more specific, well-thought out questions. One of the
successful techniques is looking for the origin of the previous-level answers. How do we
decide this behavior was fraudulent? What kind of manual tuning do our customers have
to perform? How are user engagement and recommendation engine performance
currently correlated?**

- **TRYING to understand what people want is important, trying to understand what they need is critical**

Once you feel confident enough to explain the problem in simple terms, it’s time to
wrap it up. We recommend writing down your problem understanding. Usually, it’s
several paragraphs of text, but this text will eventually become the cornerstone of your
design document. **Don’t polish it too much for now, it’s just your first (though very
important) step.**

#### 2.2.1 How can we approximate a solution through an ML System

Inexperienced or just hasty engineers may try to drag the problem into a Procrustean
bed of well-known machine learning algorithm families like supervised or unsupervised
learning, classification or regression problem. We don't think it’s the best way to start.

Imagine you got a magic oracle, a universal machine that can answer any properly
formulated question. Your job would be to approximate its behavior using machine
learning algorithms, but before mimicking it, we need to find the right question. In less
metaphoric words, here we reframe a business problem into a software/ML problem.

Some questions may seem very straightforward: 
	For the fraud problem, we want the oracle to label a user a fraudster as
	soon as possible, in the perfect world even before they did anything.
	Sounds like a sort of classification, right?

### 2.3 Risks, limitations, possible consequences

Imagine you've built a fraud detection system: it scores user activity and prevents
malicious events by suspending risky accounts. It’s a precious thing—zero fraudsters
have come through since its launch, and the Customer Success Team is happy. But
recently the Marketing Team has launched a big ad campaign, and your perfect fraud
detector banned a fair share of new users based on their traffic source (it’s unknown and
therefore somewhat suspicious according to your algorithms). Negative effects on
marketing could have been way more significant than the efficiency in detecting fraud
activity.

- **One shouldn’t think “Our team is professional, a failure like that just can’t happen here”. So, explicit thinking about risks is the way to go, as there’s a high chance of potential risks spreading beyond the project team or a single department.**

The cornerstone for any defensive strategy is a risk model.Simply put, it’s an answer
to the “What are we protecting from?” question. What are the worst scenarios possible,
and what should we avoid? Answers like “incorrect model prediction” are not informative
at all. It’s detailed understanding aligned with all possible stakeholders that is absolutely
required.

Understanding the risks and limitations will affect many future decisions, and we will
cover it later in chapters dedicated to datasets, metrics, reporting, and fallback. Before
we do though, we’d like to give a couple of examples displaying how considering (or
ignoring) valuable data can affect your goal setting.

### 2.4 Costs of a mistake

**Correctness vs. robustness**

Understanding costs of mistakes is one
of the critical points in gathering pre-design information. This is effectively a quantitative
development of the risks concept: for risks, we define what can go wrong and what we
want to avoid, and later try to assign numerical attributes.understanding costs of mistakes is one
of the critical points in gathering pre-design information. This is effectively a quantitative
development of the risks concept: for risks, we define what can go wrong and what we
want to avoid, and later try to assign numerical attributes.

