package engine

func init() {
	NewScalarValue("NUndone", "", "", func() float64 {
		return float64(NUndone)
	})
}
