# coding=utf-8

import os
import pdb
import copy
import torch
import torch.nn.functional as F
from torch import nn

from torch.nn import CrossEntropyLoss
from transformers import (
        BertConfig,
        BertModel,
        RobertaModel,
        BertForTokenClassification,
        BertTokenizer,
        RobertaConfig,
        RobertaForTokenClassification,
        RobertaTokenizer
)

class BERTMultiNER2(BertForTokenClassification):
    def __init__(self, config, num_labels=3):
        super(BERTMultiNER2, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        
        self.dise_classifier = torch.nn.Linear(config.hidden_size, self.num_labels) # disease
        self.chem_classifier = torch.nn.Linear(config.hidden_size, self.num_labels) # chemical
        self.gene_classifier = torch.nn.Linear(config.hidden_size, self.num_labels) # gene/protein
        self.spec_classifier = torch.nn.Linear(config.hidden_size, self.num_labels) # species
        self.cell_classifier = torch.nn.Linear(config.hidden_size, self.num_labels) # cell line

        self.dise_classifier_2 = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.chem_classifier_2 = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.gene_classifier_2 = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.spec_classifier_2 = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.cell_classifier_2 = torch.nn.Linear(config.hidden_size, config.hidden_size)

        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, entity_type_ids=None):
        sequence_output = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, head_mask=None)[0]
        batch_size,max_len,feat_dim = sequence_output.shape
        sequence_output = self.dropout(sequence_output)

        if entity_type_ids[0][0].item() == 0:
            '''
            Raw text data with trained parameters
            '''
            dise_sequence_output = F.relu(self.dise_classifier_2(sequence_output)) # disease logit value
            chem_sequence_output = F.relu(self.chem_classifier_2(sequence_output)) # chemical logit value
            gene_sequence_output = F.relu(self.gene_classifier_2(sequence_output)) # gene/protein logit value
            spec_sequence_output = F.relu(self.spec_classifier_2(sequence_output)) # species logit value
            cell_sequence_output = F.relu(self.spec_classifier_2(sequence_output)) # cell line logit value

            dise_logits = self.dise_classifier(dise_sequence_output) # disease logit value
            chem_logits = self.chem_classifier(chem_sequence_output) # chemical logit value
            gene_logits = self.gene_classifier(gene_sequence_output) # gene/protein logit value
            spec_logits = self.spec_classifier(spec_sequence_output) # species logit value
            cell_logits = self.cell_classifier(cell_sequence_output) # cell line logit value

            # update logit and sequence_output
            sequence_output = dise_sequence_output + chem_sequence_output + gene_sequence_output + spec_sequence_output + cell_sequence_output
            logits = (dise_logits, chem_logits, gene_logits, spec_logits, cell_logits)
        else:
            ''' 
            Train, Eval, Test with pre-defined entity type tags
            '''
            # make 1*1 conv to adopt entity type
            dise_idx = copy.deepcopy(entity_type_ids)
            chem_idx = copy.deepcopy(entity_type_ids)
            gene_idx = copy.deepcopy(entity_type_ids)
            spec_idx = copy.deepcopy(entity_type_ids)
            cell_idx = copy.deepcopy(entity_type_ids)

            dise_idx[dise_idx != 1] = 0
            chem_idx[chem_idx != 2] = 0
            gene_idx[gene_idx != 3] = 0
            spec_idx[spec_idx != 4] = 0
            cell_idx[cell_idx != 5] = 0

            dise_sequence_output = dise_idx.unsqueeze(-1) * sequence_output        
            chem_sequence_output = chem_idx.unsqueeze(-1) * sequence_output
            gene_sequence_output = gene_idx.unsqueeze(-1) * sequence_output
            spec_sequence_output = spec_idx.unsqueeze(-1) * sequence_output
            cell_sequence_output = cell_idx.unsqueeze(-1) * sequence_output

            # F.tanh or F.relu
            dise_sequence_output = F.relu(self.dise_classifier_2(dise_sequence_output)) # disease logit value
            chem_sequence_output = F.relu(self.chem_classifier_2(chem_sequence_output)) # chemical logit value
            gene_sequence_output = F.relu(self.gene_classifier_2(gene_sequence_output)) # gene/protein logit value
            spec_sequence_output = F.relu(self.spec_classifier_2(spec_sequence_output)) # species logit value
            cell_sequence_output = F.relu(self.cell_classifier_2(cell_sequence_output)) # cell line logit value

            dise_logits = self.dise_classifier(dise_sequence_output) # disease logit value
            chem_logits = self.chem_classifier(chem_sequence_output) # chemical logit value
            gene_logits = self.gene_classifier(gene_sequence_output) # gene/protein logit value
            spec_logits = self.spec_classifier(spec_sequence_output) # species logit value
            cell_logits = self.cell_classifier(cell_sequence_output) # cell line logit value

            # update logit and sequence_output
            sequence_output = dise_sequence_output + chem_sequence_output + gene_sequence_output + spec_sequence_output + cell_sequence_output
            logits = dise_logits + chem_logits + gene_logits + spec_logits + cell_logits

        outputs = (logits, sequence_output)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
                return ((loss,) + outputs)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                return loss
        else:
            return logits

class RoBERTaMultiNER2(RobertaForTokenClassification):
    def __init__(self, config, num_labels=3):
        super(RoBERTaMultiNER2, self).__init__(config)
        self.num_labels = num_labels
        self.roberta = RobertaModel(config)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        
        self.dise_classifier = torch.nn.Linear(config.hidden_size, self.num_labels) # disease
        self.chem_classifier = torch.nn.Linear(config.hidden_size, self.num_labels) # chemical
        #self.gene_classifier = torch.nn.Linear(config.hidden_size, self.num_labels) # gene/protein
        #self.spec_classifier = torch.nn.Linear(config.hidden_size, self.num_labels) # species
        self.celltype_classifier = torch.nn.Linear(config.hidden_size, self.num_labels) # cell type
        self.geneontology_classifier = torch.nn.Linear(config.hidden_size, self.num_labels) # gene ontology
        self.bodypart_classifier = torch.nn.Linear(config.hidden_size, self.num_labels) #bodypart
        self.diagnose_classifier = torch.nn.Linear(config.hidden_size, self.num_labels) #diagnose
        self.treatment_classifier = torch.nn.Linear(config.hidden_size, self.num_labels) #treatment
        #self.mutation_classifier = torch.nn.Linear(config.hidden_size, self.num_labels) #mutation
        self.location_classifier = torch.nn.Linear(config.hidden_size, self.num_labels) #location
        self.food_classifier = torch.nn.Linear(config.hidden_size, self.num_labels) #food

        self.dise_classifier_2 = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.chem_classifier_2 = torch.nn.Linear(config.hidden_size, config.hidden_size)
        #self.gene_classifier_2 = torch.nn.Linear(config.hidden_size, config.hidden_size)
        #self.spec_classifier_2 = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.celltype_classifier_2 = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.geneontology_classifier_2 = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.bodypart_classifier_2 = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.diagnose_classifier_2 = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.treatment_classifier_2 = torch.nn.Linear(config.hidden_size, config.hidden_size)
        #self.mutation_classifier_2 = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.location_classifier_2 = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.food_classifier_2 = torch.nn.Linear(config.hidden_size, config.hidden_size)

        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, entity_type_ids=None):
        sequence_output = self.roberta(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, head_mask=None)[0]
        batch_size,max_len,feat_dim = sequence_output.shape
        sequence_output = self.dropout(sequence_output)

        if entity_type_ids[0][0].item() == 0:
            '''
            Raw text data with trained parameters
            '''
            dise_sequence_output = F.relu(self.dise_classifier_2(sequence_output)) # disease logit value
            chem_sequence_output = F.relu(self.chem_classifier_2(sequence_output)) # chemical logit value
            #gene_sequence_output = F.relu(self.gene_classifier_2(sequence_output)) # gene/protein logit value
            #spec_sequence_output = F.relu(self.spec_classifier_2(sequence_output)) # species logit value
            celltype_sequence_output = F.relu(self.celltype_classifier_2(sequence_output)) # cell type logit value
            geneontology_sequence_output = F.relu(self.geneontology_classifier_2(sequence_output)) # gene ontology logit value
            bodypart_sequence_output = F.relu(self.bodypart_classifier_2(sequence_output)) # body part logit value
            diagnose_sequence_output = F.relu(self.diagnose_classifier_2(sequence_output)) # diagnose logit value
            treatment_sequence_output = F.relu(self.treatment_classifier_2(sequence_output)) # treatment logit value
            #mutation_sequence_output = F.relu(self.mutation_classifier_2(sequence_output)) # mutation logit value
            location_sequence_output = F.relu(self.location_classifier_2(sequence_output)) # location logit value
            food_sequence_output = F.relu(self.food_classifier_2(sequence_output)) # food logit value

            dise_logits = self.dise_classifier(dise_sequence_output) # disease logit value
            chem_logits = self.chem_classifier(chem_sequence_output) # chemical logit value
            #gene_logits = self.gene_classifier(gene_sequence_output) # gene/protein logit value
            #spec_logits = self.spec_classifier(spec_sequence_output) # species logit value
            celltype_logits = self.celltype_classifier(celltype_sequence_output) # cell type logit value
            geneontology_logits = self.geneontology_classifier(geneontology_sequence_output)  # gene ontology logit value
            bodypart_logits = self.bodypart_classifier(bodypart_sequence_output)  # body part logit value
            diagnose_logits = self.diagnose_classifier(diagnose_sequence_output) # diagnose logit value
            treatment_logits = self.treatment_classifier(treatment_sequence_output)# treatment logit value
            #mutation_logits = self.mutation_classifier(mutation_sequence_output)# mutation logit value
            location_logits = self.location_classifier(location_sequence_output)# mutation logit value
            food_logits = self.food_classifier(food_sequence_output)# mutation logit value


            # update logit and sequence_output
            sequence_output = dise_sequence_output + chem_sequence_output +  celltype_sequence_output + geneontology_sequence_output + bodypart_sequence_output + diagnose_sequence_output + treatment_sequence_output + location_sequence_output + food_sequence_output
            logits = (dise_logits, chem_logits, celltype_logits, geneontology_logits, bodypart_logits, diagnose_logits, treatment_logits, location_logits, food_logits)
        else:
            ''' 
            Train, Eval, Test with pre-defined entity type tags
            '''
            # make 1*1 conv to adopt entity type
            dise_idx = copy.deepcopy(entity_type_ids)
            chem_idx = copy.deepcopy(entity_type_ids)
            #gene_idx = copy.deepcopy(entity_type_ids)
            #spec_idx = copy.deepcopy(entity_type_ids)
            celltype_idx = copy.deepcopy(entity_type_ids)
            geneontology_idx = copy.deepcopy(entity_type_ids)
            bodypart_idx = copy.deepcopy(entity_type_ids)
            diagnose_idx = copy.deepcopy(entity_type_ids)
            treatment_idx = copy.deepcopy(entity_type_ids)
            #mutation_idx = copy.deepcopy(entity_type_ids)
            location_idx = copy.deepcopy(entity_type_ids)
            food_idx = copy.deepcopy(entity_type_ids)

            dise_idx[dise_idx != 1] = 0
            chem_idx[chem_idx != 2] = 0
            #gene_idx[gene_idx != 3] = 0
            #spec_idx[spec_idx != 4] = 0
            celltype_idx[celltype_idx != 5] = 0
            geneontology_idx[geneontology_idx != 6] = 0
            bodypart_idx[bodypart_idx != 7] = 0
            diagnose_idx[diagnose_idx != 8] = 0
            treatment_idx[treatment_idx != 9] = 0
            #mutation_idx[mutation_idx != 10] = 0
            location_idx[location_idx != 11] = 0
            food_idx[food_idx != 12] = 0

            

            dise_sequence_output = dise_idx.unsqueeze(-1) * sequence_output        
            chem_sequence_output = chem_idx.unsqueeze(-1) * sequence_output
            #gene_sequence_output = gene_idx.unsqueeze(-1) * sequence_output
            #spec_sequence_output = spec_idx.unsqueeze(-1) * sequence_output
            celltype_sequence_output = celltype_idx.unsqueeze(-1) * sequence_output
            geneontology_sequence_output = geneontology_idx.unsqueeze(-1) * sequence_output
            bodypart_sequence_output = bodypart_idx.unsqueeze(-1) * sequence_output
            diagnose_sequence_output = diagnose_idx.unsqueeze(-1) * sequence_output
            treatment_sequence_output = treatment_idx.unsqueeze(-1) * sequence_output
            #mutation_sequence_output = mutation_idx.unsqueeze(-1) * sequence_output
            location_sequence_output = location_idx.unsqueeze(-1) * sequence_output
            food_sequence_output = food_idx.unsqueeze(-1) * sequence_output

            # F.tanh or F.relu
            dise_sequence_output = F.relu(self.dise_classifier_2(dise_sequence_output)) # disease logit value
            chem_sequence_output = F.relu(self.chem_classifier_2(chem_sequence_output)) # chemical logit value
            #gene_sequence_output = F.relu(self.gene_classifier_2(gene_sequence_output)) # gene/protein logit value
            #spec_sequence_output = F.relu(self.spec_classifier_2(spec_sequence_output)) # species logit value
            celltype_sequence_output = F.relu(self.celltype_classifier_2(celltype_sequence_output)) # cell type logit value
            geneontology_sequence_output = F.relu(self.geneontology_classifier_2(geneontology_sequence_output)) # gene ontology logit value
            bodypart_sequence_output = F.relu(self.bodypart_classifier_2(bodypart_sequence_output)) # body part logit value
            diagnose_sequence_output = F.relu(self.diagnose_classifier_2(diagnose_sequence_output)) # diagnose logit value
            treatment_sequence_output = F.relu(self.treatment_classifier_2(treatment_sequence_output)) # treatment logit value
            #mutation_sequence_output = F.relu(self.mutation_classifier_2(mutation_sequence_output)) # mutation logit value
            location_sequence_output = F.relu(self.location_classifier_2(location_sequence_output)) # location logit value
            food_sequence_output = F.relu(self.food_classifier_2(food_sequence_output)) # food logit value

            dise_logits = self.dise_classifier(dise_sequence_output) # disease logit value
            chem_logits = self.chem_classifier(chem_sequence_output) # chemical logit value
            #gene_logits = self.gene_classifier(gene_sequence_output) # gene/protein logit value
            #spec_logits = self.spec_classifier(spec_sequence_output) # species logit value
            celltype_logits = self.celltype_classifier(celltype_sequence_output) # cell type logit value
            geneontology_logits = self.geneontology_classifier(geneontology_sequence_output)  # gene ontology logit value
            bodypart_logits = self.bodypart_classifier(bodypart_sequence_output)  # body part logit value
            diagnose_logits = self.diagnose_classifier(diagnose_sequence_output) # diagnose logit value
            treatment_logits = self.treatment_classifier(treatment_sequence_output)# treatment logit value
            #mutation_logits = self.mutation_classifier(mutation_sequence_output)# mutation logit value
            location_logits = self.location_classifier(location_sequence_output)# mutation logit value
            food_logits = self.food_classifier(food_sequence_output)# mutation logit value

            # update logit and sequence_output
            sequence_output = dise_sequence_output + chem_sequence_output +  celltype_sequence_output + geneontology_sequence_output + bodypart_sequence_output + diagnose_sequence_output + treatment_sequence_output +  location_sequence_output + food_sequence_output
            logits = dise_logits + chem_logits + celltype_logits + geneontology_logits + bodypart_logits + diagnose_logits + treatment_logits + location_logits + food_logits

        outputs = (logits, sequence_output)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                if entity_type_ids[0][0].item() == 0:
                    active_loss = attention_mask.view(-1) == 1
                    dise_logits, chem_logits, celltype_logits, geneontology_logits, bodypart_logits, diagnose_logits, treatment_logits,location_logits,food_logits= logits

                    active_dise_logits = dise_logits.view(-1, self.num_labels)
                    active_chem_logits = chem_logits.view(-1, self.num_labels)
                    #active_gene_logits = gene_logits.view(-1, self.num_labels)
                    #active_spec_logits = spec_logits.view(-1, self.num_labels)
                    active_celltype_logits = celltype_logits.view(-1, self.num_labels)
                    active_geneontology_logits = geneontology_logits.view(-1, self.num_labels)
                    active_bodypart_logits = bodypart_logits.view(-1, self.num_labels)
                    active_diagnose_logits = diagnose_logits.view(-1, self.num_labels)
                    active_treatment_logits = treatment_logits.view(-1, self.num_labels)
                    #active_mutation_logits = mutation_logits.view(-1, self.num_labels)
                    active_location_logits = location_logits.view(-1, self.num_labels)
                    active_food_logits = food_logits.view(-1, self.num_labels)
                    
                    active_labels = torch.where(
                        active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                    )
                    dise_loss = loss_fct(active_dise_logits, active_labels)
                    chem_loss = loss_fct(active_chem_logits, active_labels)
                    #gene_loss = loss_fct(active_gene_logits, active_labels)
                    #spec_loss = loss_fct(active_spec_logits, active_labels)
                    celltype_loss = loss_fct(active_celltype_logits, active_labels)
                    geneontology_loss = loss_fct(active_geneontology_logits, active_labels)
                    bodypart_loss = loss_fct(active_bodypart_logits, active_labels)
                    diagnose_loss = loss_fct(active_diagnose_logits, active_labels)
                    treatment_loss = loss_fct(active_treatment_logits, active_labels)
                    #mutation_loss = loss_fct(active_mutation_logits, active_labels)
                    location_loss = loss_fct(active_location_logits, active_labels)
                    food_loss = loss_fct(active_food_logits, active_labels)
                    loss = dise_loss + chem_loss + celltype_loss + geneontology_loss + bodypart_loss + diagnose_loss + treatment_loss +location_loss+food_loss

                    return ((loss,) + outputs)
                else:
                    active_loss = attention_mask.view(-1) == 1
                    active_logits = logits.view(-1, self.num_labels)
                    active_labels = torch.where(
                        active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                    )
                    loss = loss_fct(active_logits, active_labels)
                    return ((loss,) + outputs)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                return loss
        else:
            return logits


class NER(BertForTokenClassification):
    def __init__(self, config, num_labels=3):
        super(NER, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.classifier = torch.nn.Linear(config.hidden_size, self.num_labels)

        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        sequence_output = self.bert(input_ids, token_type_ids, attention_mask, head_mask=None)[0]
        batch_size,max_len,feat_dim = sequence_output.shape
        sequence_output = self.dropout(sequence_output)

        logits = self.classifier(sequence_output)

        outputs = (logits, sequence_output)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
                return ((loss,) + outputs)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                return loss
        else:
            return logits

