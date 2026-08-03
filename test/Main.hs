module Main (main) where

import Test.HUnit

import IrisTest
import MnistTest (testMnist)

allTests :: Test
allTests = TestList [testMnist]

main :: IO ()
main = do
  putStrLn "Running tests..."
  runTestTT allTests
  return ()
