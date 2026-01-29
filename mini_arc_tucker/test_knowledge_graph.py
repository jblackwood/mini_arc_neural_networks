"""Unit tests for knowledge graph encoding of ARC tasks."""

import torch
import pytest
from mini_arc_tucker.nb_tucker import (
    encode_arc_to_knowledge_graph,
    encode_arc_to_knowledge_graph_vectorized,
    decode_knowledge_graph_to_triples,
    get_knowledge_graph_dimensions,
    mask_output_grid_relations,
    mask_output_grid_relations_vectorized,
    reshape_task_embedding_to_3d,
    concatenate_kg_and_task_embedding,
    extract_kg_and_task_embedding,
    RELATION_ROW,
    RELATION_COLUMN,
    RELATION_CELL_VALUE,
    RELATION_GRID_TYPE,
    RELATION_ABOVE,
    RELATION_IDENTITY,
    OBJECT_ROW_COL_START,
    OBJECT_COLOR_START,
    OBJECT_GRID_TYPE_START,
    OBJECT_INPUT_GRID,
    OBJECT_OUTPUT_GRID,
    NUM_RELATIONS,
    NUM_OBJECTS,
)


class TestKnowledgeGraphDimensions:
    """Tests for knowledge graph dimensions."""
    
    def test_dimensions(self):
        dims = get_knowledge_graph_dimensions()
        
        assert dims.num_cells == 50
        assert dims.num_rows_cols == 5
        assert dims.num_subjects == 55
        assert dims.num_relations == 6
        assert dims.num_row_col_objects == 5
        assert dims.num_color_objects == 10
        assert dims.num_grid_type_objects == 2
        assert dims.num_objects == 17


class TestEncodeArcToKnowledgeGraph:
    """Tests for encoding ARC tasks to knowledge graph tensors."""
    
    def test_output_shape(self):
        """Test that output has correct shape."""
        batch = torch.zeros(1, 50, dtype=torch.long)
        kg_tensor = encode_arc_to_knowledge_graph(batch)
        
        dims = get_knowledge_graph_dimensions()
        assert kg_tensor.shape == (1, dims.num_subjects, dims.num_relations, dims.num_objects)
        assert kg_tensor.shape == (1, 55, 6, 17)
    
    def test_batch_size_multiple(self):
        """Test with larger batch size."""
        batch = torch.zeros(4, 50, dtype=torch.long)
        kg_tensor = encode_arc_to_knowledge_graph(batch)
        
        assert kg_tensor.shape == (4, 55, 6, 17)
    
    def test_cell_0_relations(self):
        """Test that cell_0 (first input cell) has correct relations."""
        # Create a batch with cell_0 having value 8
        batch = torch.zeros(1, 50, dtype=torch.long)
        batch[0, 0] = 8
        
        kg_tensor = encode_arc_to_knowledge_graph(batch)
        
        # cell_0 should have:
        # - row relation to 0
        # - column relation to 0
        # - cell_value relation to color_8
        # - grid_type relation to input_grid
        
        assert kg_tensor[0, 0, RELATION_ROW, OBJECT_ROW_COL_START + 0] == 1.0
        assert kg_tensor[0, 0, RELATION_COLUMN, OBJECT_ROW_COL_START + 0] == 1.0
        assert kg_tensor[0, 0, RELATION_CELL_VALUE, OBJECT_COLOR_START + 8] == 1.0
        assert kg_tensor[0, 0, RELATION_GRID_TYPE, OBJECT_INPUT_GRID] == 1.0
    
    def test_cell_26_relations(self):
        """Test that cell_26 (second output cell, first in output grid) has correct relations."""
        # Create a batch with cell_26 having value 9
        batch = torch.zeros(1, 50, dtype=torch.long)
        batch[0, 25] = 9  # First output cell
        
        kg_tensor = encode_arc_to_knowledge_graph(batch)
        
        # cell_25 (subject index 25) should have:
        # - row relation to 0
        # - column relation to 0
        # - cell_value relation to color_9
        # - grid_type relation to output_grid
        
        assert kg_tensor[0, 25, RELATION_ROW, OBJECT_ROW_COL_START + 0] == 1.0
        assert kg_tensor[0, 25, RELATION_COLUMN, OBJECT_ROW_COL_START + 0] == 1.0
        assert kg_tensor[0, 25, RELATION_CELL_VALUE, OBJECT_COLOR_START + 9] == 1.0
        assert kg_tensor[0, 25, RELATION_GRID_TYPE, OBJECT_OUTPUT_GRID] == 1.0
    
    def test_input_grid_cells_have_input_grid_type(self):
        """Test all input grid cells have input_grid grid_type."""
        batch = torch.zeros(1, 50, dtype=torch.long)
        kg_tensor = encode_arc_to_knowledge_graph(batch)
        
        for cell_idx in range(25):
            assert kg_tensor[0, cell_idx, RELATION_GRID_TYPE, OBJECT_INPUT_GRID] == 1.0
            assert kg_tensor[0, cell_idx, RELATION_GRID_TYPE, OBJECT_OUTPUT_GRID] == 0.0
    
    def test_output_grid_cells_have_output_grid_type(self):
        """Test all output grid cells have output_grid grid_type."""
        batch = torch.zeros(1, 50, dtype=torch.long)
        kg_tensor = encode_arc_to_knowledge_graph(batch)
        
        for cell_idx in range(25, 50):
            assert kg_tensor[0, cell_idx, RELATION_GRID_TYPE, OBJECT_OUTPUT_GRID] == 1.0
            assert kg_tensor[0, cell_idx, RELATION_GRID_TYPE, OBJECT_INPUT_GRID] == 0.0
    
    def test_integer_above_relations(self):
        """Test integer above relations: (1, above, 0), (2, above, 1), ..., (4, above, 3)."""
        batch = torch.zeros(1, 50, dtype=torch.long)
        kg_tensor = encode_arc_to_knowledge_graph(batch)
        
        for i in range(1, 5):
            subject_idx = 50 + i
            object_idx = OBJECT_ROW_COL_START + (i - 1)
            assert kg_tensor[0, subject_idx, RELATION_ABOVE, object_idx] == 1.0
    
    def test_integer_identity_relations(self):
        """Test integer identity relations: (0, identity, 0), (1, identity, 1), ..., (4, identity, 4)."""
        batch = torch.zeros(1, 50, dtype=torch.long)
        kg_tensor = encode_arc_to_knowledge_graph(batch)
        
        for i in range(5):
            subject_idx = 50 + i
            object_idx = OBJECT_ROW_COL_START + i
            assert kg_tensor[0, subject_idx, RELATION_IDENTITY, object_idx] == 1.0
    
    def test_row_column_values_for_grid_positions(self):
        """Test row and column values for various grid positions."""
        batch = torch.zeros(1, 50, dtype=torch.long)
        kg_tensor = encode_arc_to_knowledge_graph(batch)
        
        # Test a few specific positions in input grid
        # Cell 0: row 0, col 0
        assert kg_tensor[0, 0, RELATION_ROW, OBJECT_ROW_COL_START + 0] == 1.0
        assert kg_tensor[0, 0, RELATION_COLUMN, OBJECT_ROW_COL_START + 0] == 1.0
        
        # Cell 4: row 0, col 4
        assert kg_tensor[0, 4, RELATION_ROW, OBJECT_ROW_COL_START + 0] == 1.0
        assert kg_tensor[0, 4, RELATION_COLUMN, OBJECT_ROW_COL_START + 4] == 1.0
        
        # Cell 5: row 1, col 0
        assert kg_tensor[0, 5, RELATION_ROW, OBJECT_ROW_COL_START + 1] == 1.0
        assert kg_tensor[0, 5, RELATION_COLUMN, OBJECT_ROW_COL_START + 0] == 1.0
        
        # Cell 24: row 4, col 4
        assert kg_tensor[0, 24, RELATION_ROW, OBJECT_ROW_COL_START + 4] == 1.0
        assert kg_tensor[0, 24, RELATION_COLUMN, OBJECT_ROW_COL_START + 4] == 1.0
        
        # Output grid cell 25: row 0, col 0
        assert kg_tensor[0, 25, RELATION_ROW, OBJECT_ROW_COL_START + 0] == 1.0
        assert kg_tensor[0, 25, RELATION_COLUMN, OBJECT_ROW_COL_START + 0] == 1.0
        
        # Output grid cell 49: row 4, col 4
        assert kg_tensor[0, 49, RELATION_ROW, OBJECT_ROW_COL_START + 4] == 1.0
        assert kg_tensor[0, 49, RELATION_COLUMN, OBJECT_ROW_COL_START + 4] == 1.0


class TestDecodeKnowledgeGraphToTriples:
    """Tests for decoding knowledge graph tensors to triples."""
    
    def test_decode_single_batch(self):
        """Test decoding a single batch with known values."""
        batch = torch.zeros(1, 50, dtype=torch.long)
        batch[0, 0] = 8  # First input cell has value 8
        batch[0, 25] = 9  # First output cell has value 9
        
        kg_tensor = encode_arc_to_knowledge_graph(batch)
        triples = decode_knowledge_graph_to_triples(kg_tensor)
        
        assert len(triples) == 1  # One batch
        
        # Convert to set for easier checking
        triple_set = set(triples[0])
        
        # Check cell_0 relations
        assert ("cell_0", "row", "0") in triple_set
        assert ("cell_0", "column", "0") in triple_set
        assert ("cell_0", "cell_value", "color_8") in triple_set
        assert ("cell_0", "grid_type", "input_grid") in triple_set
        
        # Check cell_25 relations
        assert ("cell_25", "row", "0") in triple_set
        assert ("cell_25", "column", "0") in triple_set
        assert ("cell_25", "cell_value", "color_9") in triple_set
        assert ("cell_25", "grid_type", "output_grid") in triple_set
        
        # Check integer above relations
        assert ("int_1", "above", "0") in triple_set
        assert ("int_4", "above", "3") in triple_set
        
        # Check integer identity relations
        assert ("int_0", "identity", "0") in triple_set
        assert ("int_4", "identity", "4") in triple_set
    
    def test_roundtrip_encoding_decoding(self):
        """Test that encoding and then decoding produces expected triples."""
        # Create a specific batch
        batch = torch.zeros(1, 50, dtype=torch.long)
        batch[0, 0] = 1
        batch[0, 6] = 2  # row 1, col 1
        batch[0, 12] = 3  # row 2, col 2
        batch[0, 25] = 4  # output cell 0
        batch[0, 31] = 5  # output cell 6, row 1, col 1
        
        kg_tensor = encode_arc_to_knowledge_graph(batch)
        triples = decode_knowledge_graph_to_triples(kg_tensor)
        
        triple_set = set(triples[0])
        
        # Verify specific cell values
        assert ("cell_0", "cell_value", "color_1") in triple_set
        assert ("cell_6", "cell_value", "color_2") in triple_set
        assert ("cell_6", "row", "1") in triple_set
        assert ("cell_6", "column", "1") in triple_set
        assert ("cell_12", "cell_value", "color_3") in triple_set
        assert ("cell_12", "row", "2") in triple_set
        assert ("cell_12", "column", "2") in triple_set
        
        assert ("cell_25", "cell_value", "color_4") in triple_set
        assert ("cell_31", "cell_value", "color_5") in triple_set
        assert ("cell_31", "row", "1") in triple_set
        assert ("cell_31", "column", "1") in triple_set
    
    def test_decode_counts_expected_triples(self):
        """Test that we get the expected number of triples."""
        batch = torch.zeros(1, 50, dtype=torch.long)
        kg_tensor = encode_arc_to_knowledge_graph(batch)
        triples = decode_knowledge_graph_to_triples(kg_tensor)
        
        # Expected triples:
        # - 50 cells * 4 relations (row, column, cell_value, grid_type) = 200 triples
        # - 4 integers with above relation (1-4) = 4 triples
        # - 5 integers with identity relation (0-4) = 5 triples
        # Total = 209 triples
        
        expected_count = 50 * 4 + 4 + 5
        assert len(triples[0]) == expected_count


class TestVectorizedEncoding:
    """Tests for vectorized encoding function."""
    
    def test_vectorized_matches_loop(self):
        """Test that vectorized encoding matches loop-based encoding."""
        torch.manual_seed(42)
        batch = torch.randint(0, 10, (4, 50), dtype=torch.long)
        
        kg_loop = encode_arc_to_knowledge_graph(batch)
        kg_vectorized = encode_arc_to_knowledge_graph_vectorized(batch)
        
        assert torch.allclose(kg_loop, kg_vectorized)
    
    def test_vectorized_output_shape(self):
        """Test vectorized output shape."""
        batch = torch.zeros(8, 50, dtype=torch.long)
        kg_tensor = encode_arc_to_knowledge_graph_vectorized(batch)
        
        assert kg_tensor.shape == (8, 55, 6, 17)


class TestMaskOutputGridRelations:
    """Tests for masking output grid relations."""
    
    def test_mask_no_cells(self):
        """Test masking with no cells masked."""
        batch = torch.randint(0, 10, (1, 50), dtype=torch.long)
        kg_tensor = encode_arc_to_knowledge_graph(batch)
        
        cell_mask = torch.zeros(1, 25, dtype=torch.bool)
        masked = mask_output_grid_relations(kg_tensor, cell_mask)
        
        assert torch.allclose(masked, kg_tensor)
    
    def test_mask_all_cells(self):
        """Test masking with all output cells masked."""
        batch = torch.randint(0, 10, (1, 50), dtype=torch.long)
        kg_tensor = encode_arc_to_knowledge_graph(batch)
        
        cell_mask = torch.ones(1, 25, dtype=torch.bool)
        masked = mask_output_grid_relations(kg_tensor, cell_mask)
        
        # All output grid cells should be zeroed
        assert masked[:, 25:50, :, :].sum() == 0
        
        # Input grid cells should be unchanged
        assert torch.allclose(masked[:, :25, :, :], kg_tensor[:, :25, :, :])
        
        # Integer subjects should be unchanged
        assert torch.allclose(masked[:, 50:, :, :], kg_tensor[:, 50:, :, :])
    
    def test_mask_specific_cells(self):
        """Test masking specific output cells."""
        batch = torch.randint(0, 10, (1, 50), dtype=torch.long)
        kg_tensor = encode_arc_to_knowledge_graph(batch)
        
        cell_mask = torch.zeros(1, 25, dtype=torch.bool)
        cell_mask[0, 0] = True  # Mask first output cell (subject 25)
        cell_mask[0, 10] = True  # Mask cell 10 (subject 35)
        
        masked = mask_output_grid_relations(kg_tensor, cell_mask)
        
        # Masked cells should be zero
        assert masked[0, 25, :, :].sum() == 0
        assert masked[0, 35, :, :].sum() == 0
        
        # Other output cells should be unchanged
        assert masked[0, 26, :, :].sum() > 0
        assert masked[0, 30, :, :].sum() > 0
    
    def test_vectorized_mask_matches_loop(self):
        """Test that vectorized masking matches loop-based masking."""
        torch.manual_seed(42)
        batch = torch.randint(0, 10, (4, 50), dtype=torch.long)
        kg_tensor = encode_arc_to_knowledge_graph(batch)
        
        cell_mask = torch.randint(0, 2, (4, 25), dtype=torch.bool)
        
        masked_loop = mask_output_grid_relations(kg_tensor, cell_mask)
        masked_vectorized = mask_output_grid_relations_vectorized(kg_tensor, cell_mask)
        
        assert torch.allclose(masked_loop, masked_vectorized)


class TestTaskEmbeddingReshape:
    """Tests for task embedding reshaping."""
    
    def test_reshape_with_exact_fit(self):
        """Test reshaping when embedding fits exactly."""
        embedding = torch.randn(2, 27)  # 27 = 3*3*3
        reshaped = reshape_task_embedding_to_3d(embedding, (3, 3, 3))
        
        assert reshaped.shape == (2, 3, 3, 3)
        # Check values match when flattened
        assert torch.allclose(reshaped.view(2, -1)[:, :27], embedding)
    
    def test_reshape_with_padding(self):
        """Test reshaping with padding needed."""
        embedding = torch.randn(2, 20)  # 20 < 27 = 3*3*3
        reshaped = reshape_task_embedding_to_3d(embedding, (3, 3, 3))
        
        assert reshaped.shape == (2, 3, 3, 3)
        # Check original values preserved
        assert torch.allclose(reshaped.view(2, -1)[:, :20], embedding)
        # Check padding is zeros
        assert torch.allclose(reshaped.view(2, -1)[:, 20:], torch.zeros(2, 7))
    
    def test_reshape_truncation_raises_error(self):
        """Test that truncation raises an error."""
        embedding = torch.randn(2, 30)  # 30 > 27 = 3*3*3
        
        with pytest.raises(AssertionError):
            reshape_task_embedding_to_3d(embedding, (3, 3, 3))


class TestConcatenateKgAndTaskEmbedding:
    """Tests for concatenating KG tensor and task embedding."""
    
    def test_concatenation_shape(self):
        """Test that concatenation produces correct shape."""
        kg_tensor = torch.randn(2, 55, 6, 17)
        task_embedding_3d = torch.randn(2, 4, 4, 5)
        
        combined = concatenate_kg_and_task_embedding(kg_tensor, task_embedding_3d)
        
        assert combined.shape == (2, 59, 10, 22)
    
    def test_concatenation_preserves_values(self):
        """Test that concatenation preserves original values."""
        kg_tensor = torch.randn(2, 55, 6, 17)
        task_embedding_3d = torch.randn(2, 4, 4, 5)
        
        combined = concatenate_kg_and_task_embedding(kg_tensor, task_embedding_3d)
        
        # KG tensor should be in the first part
        assert torch.allclose(combined[:, :55, :6, :17], kg_tensor)
        
        # Task embedding should be in the extended part
        assert torch.allclose(combined[:, 55:, 6:, 17:], task_embedding_3d)
    
    def test_extraction_roundtrip(self):
        """Test that extraction recovers original tensors."""
        kg_tensor = torch.randn(2, 55, 6, 17)
        task_embedding_3d = torch.randn(2, 4, 4, 5)
        
        combined = concatenate_kg_and_task_embedding(kg_tensor, task_embedding_3d)
        
        kg_extracted, te_extracted = extract_kg_and_task_embedding(
            combined,
            kg_shape=(55, 6, 17),
            te_shape=(4, 4, 5),
        )
        
        assert torch.allclose(kg_extracted, kg_tensor)
        assert torch.allclose(te_extracted, task_embedding_3d)


class TestIntegration:
    """Integration tests for the full encoding pipeline."""
    
    def test_full_pipeline_batch_size_1(self):
        """Test full pipeline with batch_size=1 and verify correctness."""
        # Create a simple known batch
        # Input grid: all zeros except cell 0 = 8
        # Output grid: all zeros except cell 25 = 9
        batch = torch.zeros(1, 50, dtype=torch.long)
        batch[0, 0] = 8
        batch[0, 25] = 9
        
        # Encode to knowledge graph
        kg_tensor = encode_arc_to_knowledge_graph(batch)
        
        # Verify shape
        assert kg_tensor.shape == (1, 55, 6, 17)
        
        # Decode to triples
        triples = decode_knowledge_graph_to_triples(kg_tensor)
        triple_set = set(triples[0])
        
        # Verify expected triples for cell_0
        expected_cell_0 = [
            ("cell_0", "row", "0"),
            ("cell_0", "column", "0"),
            ("cell_0", "cell_value", "color_8"),
            ("cell_0", "grid_type", "input_grid"),
        ]
        for t in expected_cell_0:
            assert t in triple_set, f"Missing triple: {t}"
        
        # Verify expected triples for cell_25 (first output cell)
        expected_cell_25 = [
            ("cell_25", "row", "0"),
            ("cell_25", "column", "0"),
            ("cell_25", "cell_value", "color_9"),
            ("cell_25", "grid_type", "output_grid"),
        ]
        for t in expected_cell_25:
            assert t in triple_set, f"Missing triple: {t}"
        
        # Verify integer relations
        assert ("int_1", "above", "0") in triple_set
        assert ("int_4", "above", "3") in triple_set
        assert ("int_0", "identity", "0") in triple_set
        assert ("int_4", "identity", "4") in triple_set
        
        # Create task embedding and combine
        task_embedding = torch.randn(1, 32)
        task_embedding_3d = reshape_task_embedding_to_3d(task_embedding, (4, 4, 3))
        
        combined = concatenate_kg_and_task_embedding(kg_tensor, task_embedding_3d)
        
        # Verify combined shape
        assert combined.shape == (1, 59, 10, 20)
        
        # Extract and verify
        kg_extracted, te_extracted = extract_kg_and_task_embedding(
            combined,
            kg_shape=(55, 6, 17),
            te_shape=(4, 4, 3),
        )
        
        assert torch.allclose(kg_extracted, kg_tensor)
        assert torch.allclose(te_extracted, task_embedding_3d)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
