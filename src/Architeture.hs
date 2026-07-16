module Architeture where

type Dimension                  = Int
type Gradient                   = Double
type Input                      = Double
type Result                     = Double
type Target                     = Double
type Weight                     = Double

type Dimensions                 = [Dimension]
type Gradients                  = ([Gradient], Dimensions)
type Inputs                     = ([Input],  Dimensions)
type Results                    = ([Result], Dimensions)
type Targets                    = ([Target], Dimensions)
type Weights                    = ([Weight], Dimensions)

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

convolution :: Inputs -> Inputs -> [Double]
convolution (array, [dimensionArray]) (kernel, [dimensionKernel]) =
    [ (sum . zipWith (*) ( ( take dimensionKernel . drop diffDimensions ) array )) kernel 
    | diffDimensions <- [0 .. dimensionArray - dimensionKernel] ]
convolution (array, (dimensionArray:dimensionsArray)) (kernel, (dimensionKernel:dimensionsKernel)) =
    concat [ convolution' cutArray | cutArray <- [0 .. dimensionArray - dimensionKernel] ]
  where
    cutDimensionArray     = product dimensionsArray 
    cutDimensionKernel    = product dimensionsKernel
    convolution' cutArray = foldl1 (zipWith (+))
        [ convolution ((drop ((cutArray + cutKernel) * cutDimensionArray) array), dimensionsArray) 
                      ((take cutDimensionKernel (drop (cutKernel * cutDimensionKernel) kernel)), dimensionsKernel) 
        | cutKernel <- [0 .. dimensionKernel - 1] ]
convolution _ _ = []

padding :: Inputs -> Dimensions -> Inputs
padding (array, []) [] = (array, [])
padding (array, [d]) [p] = 
    (replicate p 0 ++ array ++ replicate p 0, [d + 2 * p])
padding (array, (d:ds)) (p:ps) = 
    (finalPad ++ concatMap fst paddeds ++ finalPad, (d + 2 * p) : snd (head paddeds))
  where
    cutDimensional = product ds
    paddeds        = [ padding (take cutDimensional (drop (i * cutDimensional) array), ds) ps | i <- [0 .. d - 1] ]
    finalPad       = replicate (p * product (snd (head paddeds))) 0
padding inputs _ = inputs

convolutionWrapper :: Inputs -> Inputs -> Dimensions -> Results
convolutionWrapper (array, dimensionsArray) (kernel, dimensionsKernel) paddings = 
    (convolution (padded, dimensionsPadded) (kernel, dimensionsKernel), (zipWith (-) dimensionsPadded . map succ) dimensionsKernel)
    where
        (padded, dimensionsPadded) = padding (array, dimensionsArray) paddings

flipN :: Inputs -> Inputs
flipN (array, dimensions) = (reverse array, dimensions)