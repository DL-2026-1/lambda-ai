module Architeture where

type Dimension                  = Int
type Gradient                   = Double
type Input                      = Double
type Result                     = Double
type Target                     = Double

type Dimensions                 = [Dimension]
type Gradients                  = [Gradient]
type Inputs                     = ([Input],  Dimensions)
type Results                    = ([Result], Dimensions)
type Targets                    = [Target]

type GradientsAll               = [Gradients]
type InputsAll                  = [Inputs]
type ResultsAll                 = [Results]

type Forward                    = (Inputs -> ResultsAll)
type Forward'                   = (Inputs -> Results)
type Forward''                  = (Inputs -> Result)

type Backward                   = (Inputs    -> Targets     -> ResultsAll -> GradientsAll)
type BackwardLast               = (Targets   -> Results     -> Gradients)
type BackwardLast'              = (Target    -> Result      -> Gradient)
type BackwardInit               = (Gradients -> InputsAll   -> GradientsAll)
type BackwardInit'              = (Gradients -> Inputs      -> Gradients)
type BackwardInit''             = (Gradients -> Inputs      -> Gradient)

type Compiler                   = (GradientsAll -> (Forward     , Backward))
type CompilerLast               = (Gradients    -> (Forward'    , BackwardLast))
type CompilerLast'              = (Gradients    -> (Forward''   , BackwardLast'))
type CompilerInit               = (GradientsAll -> (Forward     , BackwardInit))
type CompilerInit'              = (Gradients    -> (Forward'    , BackwardInit'))
type CompilerInit''             = (Gradient     -> (Forward''   , BackwardInit''))

type DeepLearnModel             = (Forward, Backward, Compiler)
type DeepLearnLayerLast         = (Forward', BackwardLast, CompilerLast)
type DeepLearnNodeLast          = (Forward'', BackwardLast', CompilerLast')
type DeepLearnModelInit         = (Forward, BackwardInit, CompilerInit)
type DeepLearnModelInitLayer    = (Forward', BackwardInit', CompilerInit')
type DeepLearnModelInitNode     = (Forward'', BackwardInit'', CompilerInit'')

convolucao :: Num a => [Int] -> [Int] -> [a] -> [a] -> [a]
convolucao [dimensaoArray] [dimensaoKernel] array kernel =
    [ (sum . zipWith (*) ( ( take dimensaoKernel . drop diferencaDimensoes ) array )) kernel 
    | diferencaDimensoes <- [0 .. dimensaoArray - dimensaoKernel] ]
convolucao (dimensaoArray:dimensoesArray) (dimensaoKernel:dimensoesKernel) array kernel =
    concat [ convolucao' fatiaArray | fatiaArray <- [0 .. dimensaoArray - dimensaoKernel] ]
  where
    recorteDimensionalA = product dimensoesArray 
    recorteDimensionalK = product dimensoesKernel
    convolucao' fatiaArray = foldl1 (zipWith (+))
        [ convolucao dimensoesArray dimensoesKernel 
            (drop ((fatiaArray + fatiaKernel) * recorteDimensionalA) array)                  
            (take recorteDimensionalK (drop (fatiaKernel * recorteDimensionalK) kernel))
        | fatiaKernel <- [0 .. dimensaoKernel - 1] ]
convolucao _ _ _ _ = []