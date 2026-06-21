module Utils where

aproxEq :: Double -> Double -> Bool
aproxEq a b = abs (a - b) < 1e-6