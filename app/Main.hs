module Main where

import Layers (MetaLayer(..), genLayer')
import Convolutional (ConvolutionalLayer'(..))
import ActivationFunction (ActivationFunction'(..))
import Model (Model(Model), train, forward)
import Utils (readCSV)
import LossFunctions (mseLF)
import Architeture (Inputs, Targets, Dataset, ResultsAll)

import Data.List (maximumBy)
import Data.Ord (comparing)

import GHC.IO (unsafePerformIO)
import GHC.Conc (getNumProcessors, getNumCapabilities, setNumCapabilities)
import System.Environment (getArgs)

configureConcurrency :: IO ()
configureConcurrency = do
  setNumCapabilities 4
  procs <- getNumProcessors
  caps  <- getNumCapabilities
  putStrLn $ "CPUs detectadas pelo SO/Docker: " ++ show procs
  putStrLn $ "Threads em uso no IHaskell:    " ++ show caps

dataTrain :: String
dataTrain      = "data/train.csv"
dataTest :: String
dataTest       = "data/test.csv"
dataValidation :: String
dataValidation = "data/validation.csv"

numeroClasses :: Int
numeroClasses = 4
numeroCanais :: Int
numeroCanais  = 22
numeroFiltros :: Int
numeroFiltros = 2
tamanhoKernel :: Int
tamanhoKernel = 50

primeiraDimensao :: Int
primeiraDimensao = 22
segundaDimensao :: Int
segundaDimensao  = 141
dimensoesEntrada :: [Int]
dimensoesEntrada = [primeiraDimensao, segundaDimensao]

quantidadeBatches :: Int
quantidadeBatches = 16

getLabel :: [Double] -> Double
getLabel = last

tiraLabel :: [Double] -> [Double]
tiraLabel = init

paraOneHot :: Double -> [Double]
paraOneHot i = [ if i == (fromIntegral j) then 1 else 0 | j <- [0..pred numeroClasses]]

preProcessaLinha :: [Double] -> (Inputs, Targets)
preProcessaLinha linha = ((tiraLabel linha, dimensoesEntrada), ((paraOneHot . getLabel) linha, [4, 1, 1, 1, 1, 1]))

generateDataset :: String -> IO Dataset
generateDataset datasetPath = do
    csv <- readCSV datasetPath
    return $ (filter ( (>0) . sum. fst. snd) . map preProcessaLinha) csv

trainSet :: Dataset
trainSet       = (unsafePerformIO .  generateDataset) dataTrain

testSet :: Dataset
testSet        = (unsafePerformIO .  generateDataset) dataTest

validationSet :: Dataset
validationSet  = (unsafePerformIO .  generateDataset) dataValidation

eegMetaLayers :: [MetaLayer]
eegMetaLayers = 
    [
         MetaConvolution (ConvolutionalLayer' { learnRate' = 0.05, filters = numeroFiltros, dimensions = [1, tamanhoKernel] }), 
         -- (c,s)x(1,t)=(c,s-t+1)=>(f,c,s-t+1)
          MetaConvolution (ConvolutionalLayer' { learnRate' = 0.05, filters = numeroFiltros, dimensions = [1, numeroCanais, 1] }),
         -- (f,c,s-t+1)x(1,c,1)=(f,1,s-t+1)=>(f,f,1,s-t+1)
          MetaActivation (ActivationFunction' "leakyRelu"),
          MetaConvolution (ConvolutionalLayer' { learnRate' = 0.05, filters = 1, dimensions = [1, 1, 1, 3] }),
         -- (f,f,1,s-t+1)x(1,1,1,3)=(f,f,1,s-t-1)=>(1,f,f,1,s-t-1)
         MetaConvolution (ConvolutionalLayer' { learnRate' = 0.05, filters = numeroClasses, dimensions = [1, numeroFiltros, numeroFiltros, 1, segundaDimensao - tamanhoKernel - 1] }),
         -- (1,f,f,1,s-t-1)x(1,f,f,1,s-t-1)=(1,1,1,1,1)=>(cls,1,1,1,1,1)
         MetaActivation (ActivationFunction' "sigmoid")
    ]

eegModel :: IO Model
eegModel = Model <$> mapM genLayer' eegMetaLayers

treinarModelo :: Int -> IO Model
treinarModelo epocas = do
    modeloInicial <- eegModel 
    return $ train modeloInicial trainSet mseLF (epocas, quantidadeBatches)

inferirTestes :: Dataset -> Model -> ResultsAll
inferirTestes testSet' modeloTreinado' =
    map ((last . forward modeloTreinado') . fst) testSet'

argMax :: Ord a => [a] -> Int
argMax lista = fst $ maximumBy (comparing snd) (zip [0..] lista)

pegarResultados :: ResultsAll -> [Int]
pegarResultados = map (argMax . fst) 

resultadosTestSet :: [Int]
resultadosTestSet = (pegarResultados . map snd) testSet

main :: IO ()
main = do
  args <- getArgs
  let epocas = if null args then 100 else read (head args) :: Int
  
  configureConcurrency
  putStrLn $ "\n>>> Iniciando treinamento com " ++ show epocas ++ " epocas <<<"
  
  modeloTr <- treinarModelo epocas
  
  let resultadosTestes = pegarResultados $ inferirTestes testSet modeloTr
      todosResultadosTexto = zipWith (\inferencia label -> show inferencia ++ "," ++ show label) resultadosTestes resultadosTestSet
      
      arquivoInferencia = "data/predicao_definitiva_" ++ show epocas ++ "epocas.csv"
      
  writeFile arquivoInferencia (unlines todosResultadosTexto)
  putStrLn $ ">>> Inferencia concluida com sucesso! Resultados salvos em: " ++ arquivoInferencia