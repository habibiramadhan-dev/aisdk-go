// internal/schema/schema_test.go
package schema_test

import (
	"encoding/json"
	"testing"

	"github.com/habibiramadhan-dev/aisdk-go/internal/schema"
)

type simplePerson struct {
	Name string `json:"name"`
	Age  int    `json:"age,omitempty"`
}

func TestReflect_ProducesFlatObjectSchema(t *testing.T) {
	raw, err := schema.Reflect[simplePerson]()
	if err != nil {
		t.Fatalf("Reflect returned error: %v", err)
	}

	var doc map[string]any
	if err := json.Unmarshal(raw, &doc); err != nil {
		t.Fatalf("Reflect output is not valid JSON: %v\n%s", err, raw)
	}

	if doc["type"] != "object" {
		t.Errorf(`doc["type"] = %v, want "object"`, doc["type"])
	}
	if _, hasRef := doc["$ref"]; hasRef {
		t.Error(`doc has a "$ref" key, want a fully-inlined schema`)
	}
	if _, hasDefs := doc["$defs"]; hasDefs {
		t.Error(`doc has a "$defs" key, want a fully-inlined schema`)
	}
	if _, hasSchemaKey := doc["$schema"]; hasSchemaKey {
		t.Error(`doc has a "$schema" key, want it stripped`)
	}
	if _, hasID := doc["$id"]; hasID {
		t.Error(`doc has a "$id" key, want it stripped`)
	}
	if doc["additionalProperties"] != false {
		t.Errorf(`doc["additionalProperties"] = %v, want false`, doc["additionalProperties"])
	}

	properties, ok := doc["properties"].(map[string]any)
	if !ok {
		t.Fatalf(`doc["properties"] = %+v, want an object`, doc["properties"])
	}
	if _, ok := properties["name"]; !ok {
		t.Error(`doc["properties"] has no "name" key`)
	}
	if _, ok := properties["age"]; !ok {
		t.Error(`doc["properties"] has no "age" key`)
	}

	required, ok := doc["required"].([]any)
	if !ok || len(required) != 1 || required[0] != "name" {
		t.Errorf(`doc["required"] = %+v, want ["name"] (Age has omitempty, so it's excluded)`, doc["required"])
	}
}

type addressBook struct {
	Owner     string    `json:"owner"`
	Addresses []address `json:"addresses"`
}

type address struct {
	City string `json:"city"`
}

func TestReflect_InlinesNestedStructsAndSlices(t *testing.T) {
	raw, err := schema.Reflect[addressBook]()
	if err != nil {
		t.Fatalf("Reflect returned error: %v", err)
	}

	var doc map[string]any
	if err := json.Unmarshal(raw, &doc); err != nil {
		t.Fatalf("Reflect output is not valid JSON: %v\n%s", err, raw)
	}
	if _, hasDefs := doc["$defs"]; hasDefs {
		t.Error(`doc has a "$defs" key, want the nested "address" struct fully inlined instead`)
	}

	properties := doc["properties"].(map[string]any)
	addresses, ok := properties["addresses"].(map[string]any)
	if !ok {
		t.Fatalf(`properties["addresses"] = %+v, want an object`, properties["addresses"])
	}
	items, ok := addresses["items"].(map[string]any)
	if !ok {
		t.Fatalf(`properties["addresses"]["items"] = %+v, want an object`, addresses["items"])
	}
	if _, hasRef := items["$ref"]; hasRef {
		t.Error(`properties["addresses"]["items"] has a "$ref" key, want the address struct inlined directly`)
	}
	itemProps, ok := items["properties"].(map[string]any)
	if !ok {
		t.Fatalf(`items["properties"] = %+v, want an object`, items["properties"])
	}
	if _, ok := itemProps["city"]; !ok {
		t.Error(`items["properties"] has no "city" key — nested struct wasn't inlined correctly`)
	}
}
