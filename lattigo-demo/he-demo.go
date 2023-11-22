package main

import (
	"fmt"
	"github.com/ldsec/lattigo/v2/ckks"
	"math/cmplx"
)

func main() {
	params := ckks.DefaultParams[ckks.PN12QP109]

	kgen := ckks.NewKeyGenerator(params)
	sk, pk := kgen.GenKeyPair()

	encryptor := ckks.NewEncryptor(params, pk)
	decryptor := ckks.NewDecryptor(params, sk)

	encoder := ckks.NewEncoder(params)

	// Create an array of complex numbers from 1 to 10
	values := make([]complex128, 10)
	for i := range values {
		values[i] = complex(float64(i+1), 0)
	}

	ciphertexts := make([]*ckks.Ciphertext, len(values))
	for i, val := range values {
		values[i] = val
		plaintext := encoder.EncodeNew(val, params.LogN())
		ciphertexts[i] = encryptor.EncryptNew(plaintext)
	}

	diffs := make([][]*ckks.Ciphertext, len(values))
	for i := range diffs {
		diffs[i] = make([]*ckks.Ciphertext, len(values))
	}

	evaluator := ckks.NewEvaluator(params)
	for i := range values {
		for j := range values {
			// Subtract the ciphertexts
			diff := evaluator.SubNew(ciphertexts[i], ciphertexts[j])

			// Take the absolute value
			absDiff := evaluator.AbsNew(diff)

			// Store the result
			diffs[i][j] = absDiff
		}
	}

	for i := range diffs {
		for j := range diffs[i] {
			decrypted := decryptor.DecryptNew(diffs[i][j])
			decoded, _ := encoder.Decode(decrypted)
			fmt.Printf("The absolute difference between %d and %d is: %f\n", i+1, j+1, cmplx.Abs(decoded[0]))
		}
	}
}
