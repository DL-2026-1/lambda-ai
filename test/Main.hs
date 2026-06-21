module Main (main) where

import PerceptronTests (testAllPerceptron)

import Test.HUnit

allTests :: Test
allTests = TestList [TestLabel "Perceptron Tests" testAllPerceptron]

main :: IO ()
main = do
  putStrLn "Running tests..."
  runTestTT allTests
  return ()
