module MyLib (main) where

import Math.NumberTheory.Primes
import Math.NumberTheory.Moduli
import Math.NumberTheory.ArithmeticFunctions
import Data.Maybe
import Numeric.Natural
import Data.Mod
import Debug.Trace
import Control.Monad
import Data.List (tails, sort)
import qualified Data.Map as M

findPrime :: Natural -> Int -> Prime Natural
findPrime m np = go 1
  where ps = take np primes
        base = product (map unPrime ps)
        go k = maybe (go (k+1)) id (isPrime $ k*base*m + 1)

{-# SCC primitiveMthRoot #-}
primitiveMthRoot :: Natural -> Prime Natural -> SomeMod
primitiveMthRoot m p = head
  [ g
  | a <- map (`modulo` unPrime p) [2..]
  , let g = powSomeMod a ((unPrime p - 1) `div` m)
  , g /= 1
  , all ((/= 1) . powSomeMod g . (m `div`)) mfacts
  ]
  where mfacts = map (unPrime . fst) (factorise m)

numRoots m p = gcd m (unPrime p - 1)
numPrimitiveRoots m p
  | (unPrime p - 1) `mod` m == 0 = totient m
  | otherwise = 0

f n m w xs = fstTrm * sndTrm
  where fstTrm = product [ (powSomeMod xj (-1)) * xk + xj * powSomeMod xk (-1)
                         | (j:ks) <- tails [0..n-1]
                         , k <- ks
                         , let xj = xs !! j
                         , let xk = xs !! k ]
        sndTrm = compDropDet n m w xs

compDropDet n m w xs = det . dropLast $ compMat n m w xs
  where dropLast = init . map init

det [] = 1
det [[a]] = a
det ass@(as:_) =
  sum [ sign * (as !! j) * det (minor j ass)
      | j <- [0..length ass-1]
      , let sign | even j = 1
                 | otherwise = -1 ]

minor j (_:as) = map (removeAt j) as
  where removeAt n as = take n as ++ drop (n+1) as


compMat n m w xs = [ [ a j k xj xk
                     | k <- [0..n-1]
                     , let xk = xs !! k ]
                   | j <- [0..n-1]
                   , let xj = xs !! j ]
  where a j k xj xk
          | j /= k = negate $ xj * (powSomeMod xk (-1)) * (powSomeMod ((powSomeMod xj (-1)) * xk + xj * powSomeMod xk (-1)) (-1))
          | otherwise = sum [ xj * (powSomeMod xr (-1)) * (powSomeMod ((powSomeMod xj (-1)) * xr + xj * powSomeMod xr (-1)) (-1))
                            | r <- [0..n-1]
                            , r /= j
                            , let xr = xs !! r ]

e_n n p m w =
  (powSomeMod (fromIntegral m `modulo` unPrime p) (-n+1)) * sum (map (f n m w) wss)
  where wss = (++ [w^m]) . map (powSomeMod w) <$> replicateM (n-1) [0..m-1]

e_n' n p m w =
  (powSomeMod (fromIntegral m `modulo` unPrime p) (-n+1)) * sum (map ((cache M.!) . sort) wss)
  where wss = (++ [w^m]) . map (powSomeMod w) <$> replicateM (n-1) [0..m-1]
        cache = foldl' go M.empty wss
        go m ws | Just r <- M.lookup (sort ws) m = m
                | otherwise = M.insert (sort ws) (f n m w ws) m

main'' :: IO ()
main'' = do
  let m = 39
  let p = findPrime m 300
  print p
  let r = primitiveMthRoot m p
  print r
  print (r^2)
  print (r^m)
  print (r^(2*m))

mFor n = head $ filter odd [(n+1) `div` 2, (n+3) `div` 2]

main' :: IO ()
main' = do
  let (Just p) = isPrime 271
  let m = 3
  let w = primitiveMthRoot m p
  print (e_n' 3 p m w)
  print (e_n' 5 p m w)

main :: IO ()
main = do
  let n = 7
  let m = mFor n
  print m
  let p = findPrime m 300
  print p
  let w = primitiveMthRoot m p
  print w
  print (e_n' (fromIntegral n) p m w)
  


-- p = 271
-- n = 3 (or 5)
-- m = 3
-- 3rd-root mod 271 = 28
-- e_n 3 = 2
-- e_n 5 = 264
