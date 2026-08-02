module Architeture where

type Dimension                  = Int
type Epoch                      = Int    
type MinBatch                   = Int

type Gradient                   = Double
type Input                      = Double
type Result                     = Double
type Target                     = Double
type Weight                     = Double

type ActivationFunction         = ((Input -> Result), (Input -> Result)) 

type Dimensions                 = [Dimension]
type Gradients                  = ([Gradient], Dimensions)
type Inputs                     = ([Input],  Dimensions)
type Results                    = ([Result], Dimensions)
type Targets                    = ([Target], Dimensions)
type Weights                    = ([Weight], Dimensions)

type Dataset                    =  [(Inputs, Targets)]

type LossFunction               = (Targets -> Results -> Weights)

type GradientsAll               = [Gradients]
type InputsAll                  = [Inputs]
type TargetsAll                 = [Targets]
type ResultsAll                 = [Results]

type Forward                    = (Inputs -> Results)
type Backward                   = (Gradients -> Inputs      -> Gradients)

type Forward'                   = (Inputs -> ResultsAll)
type Backward'                  = (Gradients -> ResultsAll -> Inputs -> GradientsAll) 

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
    (convolution (padded, dimensionsPadded) (kernel, dimensionsKernel), (zipWith (-) dimensionsPadded . map pred) dimensionsKernel)
    where
        (padded, dimensionsPadded) = padding (array, dimensionsArray) paddings

flipN :: Inputs -> Inputs
flipN (array, dimensions) = (reverse array, dimensions)