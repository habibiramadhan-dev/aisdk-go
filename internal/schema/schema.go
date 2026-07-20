// internal/schema/schema.go
package schema

import (
	"encoding/json"

	"github.com/invopop/jsonschema"
)

// Reflect produces a fully-inlined JSON Schema document for T — no $ref, no
// $defs, no $schema/$id keys — suitable for embedding directly into any of
// the three providers' structured-output request fields (Anthropic
// OutputConfig.Format, OpenAI ResponseFormat, Gemini ResponseJsonSchema).
//
// invopop/jsonschema's defaults already produce "additionalProperties":
// false and a fully-populated "required" array (every field without a
// json:",omitempty" tag) — both are exactly what OpenAI's strict mode
// requires, so no separate lowering pass runs here. A field WITH omitempty
// is correctly excluded from "required", but OpenAI strict mode actually
// wants such a field present in "required" with a nullable type instead of
// omitted — that transformation is NOT implemented; structs intended for
// GenerateStructured should avoid omitempty on fields that must always be
// present in the model's output.
//
// T must not be self-referential (directly, or via a slice/map/pointer
// field) — reflecting a recursive type causes an unrecoverable Go runtime
// stack overflow (not a catchable panic; recover() cannot stop it), because
// disabling $ref generation (DoNotReference, required for the "no recursive
// schemas" constraint every provider's structured-output path shares) also
// disables the reflector's only cycle-breaker. This is a hard, undetected
// constraint on T — recursive types are unsupported by design, not
// validated against at runtime.
func Reflect[T any]() (json.RawMessage, error) {
	reflector := &jsonschema.Reflector{
		DoNotReference: true,
		Anonymous:      true,
	}
	var zero T
	s := reflector.Reflect(&zero)
	s.Version = ""
	s.ID = ""
	return json.Marshal(s)
}
