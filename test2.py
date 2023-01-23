import json
from cls_python import CLSDecoder, CLSEncoder, FiniteCombinatoryLogic, Subtypes

from cls_python.types import Type, Omega, Constructor, Arrow, Intersection
from cls_python.subtypes import Subtypes
from cls_python.fcl import InhabitationResult, FiniteCombinatoryLogic, MultiArrow

from itertools import chain, combinations
from functools import reduce
import timeit

a: Type = Constructor("a")
b: Type = Constructor("b")

repository: dict[object, Type] = dict[object, Type]({
  "X_(a)": a,
  "X_(b->b)": Arrow(b, b),
  "X_(b->a)": Arrow(b, a),
  "X_(o->a->a)": Arrow(Omega(), Arrow(a, a))
  })
environment: dict[object, set] = dict[object, set]()
subtypes: Subtypes = Subtypes(environment)
target = a
fcl: FiniteCombinatoryLogic = FiniteCombinatoryLogic(repository, subtypes)
result: InhabitationResult = fcl.inhabit(target)

if result.check_empty(target):
    print("No inhabitants")
else:
    for tree in result[target][0:10]:
        print(tree)
        print("")

with open("C:/Users/Dudenhefner/Desktop/repo.json", "r+") as fptorepo:
    with open("C:/Users/Dudenhefner/Desktop/taxonomy.json", "r+") as fptotaxonomy:
        with open("C:/Users/Dudenhefner/Desktop/request.json", "r+") as fptorequest:
            gamma = FiniteCombinatoryLogic(
                        json.load(fptorepo, cls=CLSDecoder),
                        Subtypes(json.load(fptotaxonomy))
                    )
            target = json.load(fptorequest, cls=CLSDecoder)
            start = timeit.default_timer()
            result = gamma.inhabit(target)
            print('Time: ', timeit.default_timer() - start) 
            if result.check_empty(target):
                print("No inhabitants")
            else:
                for tree in result[target][0:2]:
                    print(tree)
                    print("")

#results:
#Combinator(seem-think-easy-level-say-stay-local-reason-reach-federal-company-right-begin-tell-black-body -> Span_parts & Weight_attributes -> LiPo_parts -> Rpi 3B_parts -> Base_parts, RPI Base Hexa v16)(Combinator(seem-think-easy-level-say-stay-local-reason-reach-federal-company-right-begin-tell-black-body, understand-political-common-reason))(Combinator(stop-open-member-research-send-stay-hot-mother-play-speak-local-year -> 4xM2.5 Dia16mm_formats -> Flat _formats & ESC_parts -> Span_parts & Weight_attributes, 90mm 16M2.5 Format Span Lightweight v7)(Combinator(stop-open-member-research-send-stay-hot-mother-play-speak-local-year, write-hard-teacher-kind))(Combinator(bring-current-program-result-see-major-area-moment -> 5mm Shaft_formats & Propeller_parts -> 4xM2.5 Dia16mm_formats, 5mm Motor v22)(Combinator(bring-current-program-result-see-major-area-moment, expect-bring-dead-word))(Combinator(talk-allow-popular-person -> 5mm Shaft_formats & Propeller_parts, Propeller CW Puller v7)(Combinator(talk-allow-popular-person, love-create-huge-house))))(Combinator(want-good-child-week -> Flat _formats & ESC_parts, HGLRC 30A  Brushless ESC v8)(Combinator(want-good-child-week, write-set-easy-moment))))(Combinator(stay-large-full-level-take-go-long-hour -> Bottom_parts -> LiPo_parts, ACE-X 1500mAh 4S Lipo Battery v11)(Combinator(stay-large-full-level-take-go-long-hour, have-short-left-kind))(Combinator(run-real-issue-word -> Bottom_parts, RPI Bottom 6 v5)(Combinator(run-real-issue-word, remember-would-full-end))))(Combinator(put-blue-left-world -> Rpi 3B_parts, Raspberry Pi 3B Plus v7)(Combinator(put-blue-left-world, love-national-physical-teacher)))
#Combinator(seem-think-easy-level-say-stay-local-reason-reach-federal-company-right-begin-tell-black-body -> Span_parts & Weight_attributes -> LiPo_parts -> Rpi 3B_parts -> Base_parts, RPI Base Hexa v16)(Combinator(seem-think-easy-level-say-stay-local-reason-reach-federal-company-right-begin-tell-black-body, understand-political-common-reason))(Combinator(stop-open-member-research-send-stay-hot-mother-play-speak-local-year -> 4xM2.5 Dia16mm_formats -> Flat _formats & ESC_parts -> Span_parts & Weight_attributes, 90mm 16M2.5 Format Span Lightweight v7)(Combinator(stop-open-member-research-send-stay-hot-mother-play-speak-local-year, write-hard-teacher-kind))(Combinator(can-look-small-country-ask-know-wrong-group-hear-wait-dark-history-stop-could-main-community -> M2.5_formats -> M3_formats & Screw_parts & Steel_attributes -> 4xM3 Dia34mm_formats & Motor_parts -> 4xM2.5 Dia16mm_formats, 16mmM3 to 34mmM3 Lightweight v12)(Combinator(can-look-small-country-ask-know-wrong-group-hear-wait-dark-history-stop-could-main-community, live-offer-entire-government))(Combinator(fall-look-likely-face -> M2.5_formats, DIN_EN_ISO_4762_M2,5x8.ipt v6)(Combinator(fall-look-likely-face, fall-great-dead-idea)))(Combinator(ask-need-simple-word -> M3_formats & Screw_parts & Steel_attributes, DIN_EN_ISO_4762_M3x8.ipt v5)(Combinator(ask-need-simple-word, talk-various-way-hour)))(Combinator(move-military-friend-country-allow-difficult-friend-others -> 5mm Shaft_formats & Propeller_parts -> 4xM3 Dia34mm_formats & Motor_parts, 5mm Sunnsky Motor v7)(Combinator(move-military-friend-country-allow-difficult-friend-others, like-hot-hour-law))(Combinator(talk-allow-popular-person -> 5mm Shaft_formats & Propeller_parts, Propeller CW Puller v7)(Combinator(talk-allow-popular-person, love-create-huge-house)))))(Combinator(want-good-child-week -> Flat _formats & ESC_parts, HGLRC 30A  Brushless ESC v8)(Combinator(want-good-child-week, write-set-easy-moment))))(Combinator(stay-large-full-level-take-go-long-hour -> Bottom_parts -> LiPo_parts, ACE-X 1500mAh 4S Lipo Battery v11)(Combinator(stay-large-full-level-take-go-long-hour, have-short-left-kind))(Combinator(run-real-issue-word -> Bottom_parts, RPI Bottom 6 v5)(Combinator(run-real-issue-word, remember-would-full-end))))(Combinator(put-blue-left-world -> Rpi 3B_parts, Raspberry Pi 3B Plus v7)(Combinator(put-blue-left-world, love-national-physical-teacher)))

