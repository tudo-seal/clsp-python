Todo
====


 - am Ende brauchen wir die `FCL.inhabit()` Methode
   - `FCL.inhabit : \*targets -> InhabitationResult`
   - Rules intern als `dict[Type, List[Combinator, List[Type]]` z.B. sigma -> C tau_1 tau_2 | D rho  
   - Memoization als `set[Type]`, Liste der besuchten Typen als early Abort
 - InhabitationMachine -> Erstmal rekursiver InhabitaionsAlgo
   - Eingabe: ein Typ (target), Felder der Klasse (FCL) sind Dictionary (s.o.), MemoizationListe (s.o)
   - Wenn Typ schon bekannt (Memoization), dann fertig, ansonsten füge zur Liste hinzu.
   - Berechne Primfaktoren/Pfade des Typen (organize)
   - Für jedes C in Gamma und jede Stelligkeit Aufruf an SetCover,
        Sets sind die Multiarrows, Elements sind Primfaktoren, Elementbeziehung ist subtyping bezüglich des Targets
     - Setcovers sind `List[List[Multiarrow]]`
     - merge + minimieren (Da können wir Jans code nutzen)
     - => `List[Multiarrow]`, mit target sind subtypen des übergebenen Typen
     - Füge dem Dictionary die Regel hinzu: target -> C tau_1 ... tau_n | C sigma_1 ... sigma_n, wobei n die gewählte Stelligkeit 
          von C ist, und tau_1...tau_n und sigma_1 ... sigma_n die sources der Multiarows 
     => rekursiv inhabit die sources
   
   Beispiele:
  
   Gamma = {
      A : a -> a
      B : a 
   }

   inhabit(a):
     * a ist nicht bekannt, memoiziere a
     * Primfaktoren von a sind [a]
     * Jedes C in Gamma mit Stelligkeit sind: {A_0, A_1, B_0}
       * A_0: [([], a -> a)] 
         * Setcover = []
         * Nichts zu tun
       * A_1: [([a], a)]
         * Setcover: [([a], a)]
         * minimierung und merge macht nichts
         * Dictionary ist nun {a : [A [a]]}
         * inhabit(a)
           * a ist bekannt, return
       * B_0: [([], a)]
         * Setcover: [([], a)]
         * minimierung und merge macht nichts
         * Dictionary ist nun {a : [A [a], B]}
      * Fertig

    Ein bisschen post processing, um das kompatibel zu bleiben.





 - CoverMachine -> SetCover
 - 
